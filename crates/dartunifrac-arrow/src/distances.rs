use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use dartunifrac_core::unifrac_from_sketches;
use rayon::prelude::*;

use crate::coo::{coo_batch, output_schema};
use crate::MarshalError;

/// Sketches narrowed to `bbits`, the width the distance kernel compares.
///
/// b-bit minwise hashing keeps the **low** bits of each 64-bit value: two
/// sketches agree in a slot when their low `bbits` agree, and the estimator's
/// error is a function of that width. Narrowing is therefore not a storage
/// optimisation that can be skipped — it is part of the estimator, and a sketch
/// set compared at the wrong width yields different distances.
pub enum Sketches {
    U16(Vec<Vec<u16>>),
    U32(Vec<Vec<u32>>),
    U64(Vec<Vec<u64>>),
}

/// Run `$body` against whichever width the sketches are, monomorphised per arm.
///
/// `unifrac_from_sketches` is generic and `#[inline]`, so each arm gets its own
/// specialised kernel with no dynamic dispatch in the hot loop. The binary
/// dispatches the same way, for the same reason.
macro_rules! per_width {
    ($sk:expr, |$rows:ident| $body:expr) => {
        match $sk {
            Sketches::U16($rows) => $body,
            Sketches::U32($rows) => $body,
            Sketches::U64($rows) => $body,
        }
    };
}

/// [`per_width!`] over two sets of one width: `$body` runs with both narrowed,
/// and two sets of different widths run `$mismatch` instead.
macro_rules! per_width_pair {
    ($a:expr, $b:expr, |$ra:ident, $rb:ident| $body:expr, $mismatch:expr) => {
        match ($a, $b) {
            (Sketches::U16($ra), Sketches::U16($rb)) => $body,
            (Sketches::U32($ra), Sketches::U32($rb)) => $body,
            (Sketches::U64($ra), Sketches::U64($rb)) => $body,
            _ => $mismatch,
        }
    };
}

impl Sketches {
    /// How many samples, i.e. how many rows the triangle has.
    fn count(&self) -> usize {
        per_width!(self, |rows| rows.len())
    }

    /// Reject what the kernel would otherwise panic on or quietly turn to NaN,
    /// and return the values per sketch: `None` for an empty set.
    ///
    /// Both checks exist because this crate sits beneath an FFI boundary.
    /// `count_mismatches` asserts equal lengths, and an assert there aborts the
    /// host process; a zero-length sketch divides by zero and yields a complete
    /// distance matrix that is silently meaningless.
    fn validate(&self) -> Result<Option<usize>, MarshalError> {
        per_width!(self, |rows| {
            let Some(first) = rows.first() else {
                return Ok(None);
            };
            let k = first.len();
            // Two or more samples are pairs of their own.
            check_values(k, rows.len() > 1)?;
            for (s, row) in rows.iter().enumerate() {
                if row.len() != k {
                    return Err(MarshalError::Sketch(format!(
                        "sketch {s} holds {} values but sketch 0 holds {k}; \
                         sketches of different lengths are not comparable",
                        row.len()
                    )));
                }
            }
            Ok(Some(k))
        })
    }

    /// The width every value was narrowed to, in bits.
    pub(crate) fn width(&self) -> u8 {
        match self {
            Sketches::U16(_) => 16,
            Sketches::U32(_) => 32,
            Sketches::U64(_) => 64,
        }
    }
}

/// Refuse sketches of no values where they would give a distance: every one
/// would be 0/0. Only an error where there are pairs, since with none there is
/// nothing to be wrong; each reader says when its sets make pairs.
fn check_values(k: usize, pairs: bool) -> Result<(), MarshalError> {
    if k == 0 && pairs {
        return Err(MarshalError::Sketch(
            "sketches hold no values, so every distance would be NaN".into(),
        ));
    }
    Ok(())
}

/// One set as a reader takes it: [`Sketches::validate`], then one id per
/// sketch. Returns the set's values per sketch, `None` when it is empty.
fn check_set(sketches: &Sketches, ids: &[i64]) -> Result<Option<usize>, MarshalError> {
    let k = sketches.validate()?;
    let n = sketches.count();
    if ids.len() != n {
        return Err(MarshalError::Sketch(format!(
            "{} ids were given for {n} sketches; they index the same samples",
            ids.len()
        )));
    }
    Ok(k)
}

/// Narrow 64-bit sketches to `bbits`, which must be 16, 32 or 64.
///
/// The binary's equivalent warns and falls back to 16 for anything else, which
/// is safe there only because clap has already normalised the value. Behind an
/// FFI boundary the caller supplies `bbits` directly, and silently halving the
/// sketch width would change every distance the run reports without failing.
pub fn trim_to_bbits(sk: Vec<Vec<u64>>, bbits: u8) -> Result<Sketches, MarshalError> {
    match bbits {
        16 => Ok(Sketches::U16(
            sk.into_iter()
                .map(|row| row.into_iter().map(|x| x as u16).collect())
                .collect(),
        )),
        32 => Ok(Sketches::U32(
            sk.into_iter()
                .map(|row| row.into_iter().map(|x| x as u32).collect())
                .collect(),
        )),
        64 => Ok(Sketches::U64(sk)),
        other => Err(MarshalError::Sketch(format!(
            "bbits {other} is not supported; use 16, 32 or 64"
        ))),
    }
}

/// Stream the condensed upper triangle of the distance matrix, in bounded tiles.
///
/// The triangle is cut into tiles of at most `side × side`, where
/// `side = isqrt(batch_rows)`, and each tile becomes one `RecordBatch` computed
/// with rayon. Tiling **both** axes is what bounds the output: a stripe of rows
/// against all `n` columns grows with `n` — at a million samples a
/// `sqrt(n)`-row stripe is ~10⁹ pairs — whereas a `side × side` tile holds at
/// most `side²  ≤ batch_rows` pairs however large `n` gets.
///
/// Tiles are produced in a fixed order and each tile's rows are written to fixed
/// offsets, so the stream is identical for any thread count. `batch_rows` sets
/// the tile side, so it moves the batch boundaries and the order of the pairs,
/// which is tile by tile rather than row-major; which pairs are emitted, and
/// every distance, are the same for any `batch_rows`.
///
/// `ids` are the caller's own sample indices — `SketchSet::kept` — so emitted
/// `i`/`j` need no second join to interpret.
///
/// `batch_rows` only frames the output; unlike `bbits` it cannot change a
/// distance, so a request of 0 is clamped to one row rather than refused —
/// matching [`coo_reader`](crate::coo_reader), which clamps the same way. Very
/// small values are legal but wasteful: the tile side is the square root, so a
/// `batch_rows` of 100 spreads 100 pairs over 10 rayon tasks and pays a whole
/// `RecordBatch` for them.
///
/// `sketches` is shared, not copied, when passed as an `Arc`: a C API handle
/// and every stream drawn from it can then hold the one sketch set, and a
/// stream stays valid after the handle is freed. A plain [`Sketches`] works
/// too, and is simply moved into a new `Arc`.
pub fn distances_reader(
    sketches: impl Into<Arc<Sketches>>,
    ids: Vec<i64>,
    weighted: bool,
    batch_rows: usize,
) -> Result<Box<dyn RecordBatchReader + Send>, MarshalError> {
    let sketches = sketches.into();
    check_set(&sketches, &ids)?;
    Ok(tiled(sketches, ids, None, weighted, batch_rows))
}

/// Stream every pair of a `rows` × `cols` rectangle, in bounded tiles.
///
/// The two sides are separate sketch sets, such as a batch of new samples and
/// a retained set of anchors, so each carries its own `ids`: `i` is always one
/// of `row_ids` and `j` one of `col_ids`. Every pair is emitted, with no
/// `i < j` rule and no special case: the same set on both sides gives the full
/// square, diagonal included, and an empty side gives no batches.
///
/// Tiling is [`distances_reader`]'s, over the whole rectangle: tiles of at most
/// `side × side` (`side = isqrt(batch_rows)`), visited row tile by row tile, so
/// the pairs never depend on `batch_rows` or the thread count, and the order
/// depends only on `batch_rows`.
///
/// A known limit: the threads share out a tile's rows, so a rectangle of few
/// rows uses them less well. With 8 threads, 16 rows × 5,000 cost 1.23× per
/// pair what 1,000 rows did. Fewer rows than threads leaves threads idle.
/// Crosses are meant for large sets on both sides, where tiles are full.
///
/// Each side is first checked in itself, as [`distances_reader`] checks its one
/// set, and an error names the side; so a malformed side is refused even
/// against an empty one. When both sides hold sketches they must also be of
/// one width and one length, with at least one value: anything else is not a
/// comparison, and core's kernel asserts equal lengths, which beneath an FFI
/// boundary would abort the host process.
///
/// Whether the two sets are *comparable*, sketched against one tree with one
/// set of parameters by one build, cannot be told from the sketches, so a Rust
/// caller must ensure it. The C API's `df_distances_cross` does, by refusing
/// two sets whose fingerprints differ.
pub fn cross_distances_reader(
    rows: impl Into<Arc<Sketches>>,
    row_ids: Vec<i64>,
    cols: impl Into<Arc<Sketches>>,
    col_ids: Vec<i64>,
    weighted: bool,
    batch_rows: usize,
) -> Result<Box<dyn RecordBatchReader + Send>, MarshalError> {
    let (rows, cols) = (rows.into(), cols.into());
    let side = |name: &str, sketches: &Sketches, ids: &[i64]| {
        check_set(sketches, ids).map_err(|e| match e {
            MarshalError::Sketch(m) => MarshalError::Sketch(format!("{name}: {m}")),
            other => other,
        })
    };
    let kr = side("rows", &rows, &row_ids)?;
    let kc = side("cols", &cols, &col_ids)?;
    if let (Some(kr), Some(kc)) = (kr, kc) {
        if rows.width() != cols.width() {
            return Err(MarshalError::Sketch(format!(
                "rows hold {}-bit sketches but cols hold {}-bit ones; \
                 sketches of different widths are not comparable",
                rows.width(),
                cols.width()
            )));
        }
        if kr != kc {
            return Err(MarshalError::Sketch(format!(
                "rows hold sketches of {kr} values but cols hold {kc}; \
                 sketches of different lengths are not comparable"
            )));
        }
        // One sample a side is already a pair.
        check_values(kr, true)?;
    }
    Ok(tiled(rows, row_ids, Some((cols, col_ids)), weighted, batch_rows))
}

/// The reader both entry points share, over sets already validated.
fn tiled(
    rows: Arc<Sketches>,
    row_ids: Vec<i64>,
    cols: Option<(Arc<Sketches>, Vec<i64>)>,
    weighted: bool,
    batch_rows: usize,
) -> Box<dyn RecordBatchReader + Send> {
    // Load-bearing, not cosmetic: `div_ceil(0)` below panics, and a panic
    // beneath an FFI boundary aborts the host process.
    let side = batch_rows.isqrt().max(1);
    let n_rows = row_ids.len();
    let n_cols = cols.as_ref().map_or(n_rows, |(_, ids)| ids.len());
    Box::new(TiledReader {
        rows,
        row_ids,
        cols,
        weighted,
        n_rows,
        n_cols,
        side,
        row_tiles: n_rows.div_ceil(side),
        col_tiles: n_cols.div_ceil(side),
        ti: 0,
        tj: 0,
        schema: Arc::new(output_schema()),
        done: false,
    })
}

struct TiledReader {
    rows: Arc<Sketches>,
    row_ids: Vec<i64>,
    /// The column set, or `None` for the condensed upper triangle of `rows`
    /// against itself. Holding no second set is what makes the triangle: it
    /// cannot be asked for over two different sets.
    cols: Option<(Arc<Sketches>, Vec<i64>)>,
    weighted: bool,
    n_rows: usize,
    n_cols: usize,
    side: usize,
    row_tiles: usize,
    col_tiles: usize,
    ti: usize,
    tj: usize,
    schema: SchemaRef,
    done: bool,
}

/// One tile: rows `r0..r1` against columns `c0..c1`.
struct Tile {
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    /// On the diagonal, a row's columns start just past itself rather than at
    /// `c0`, which is what keeps `(i, i)` and the lower half out of the output.
    diagonal: bool,
}

impl Tile {
    /// Where row `i`'s columns begin.
    fn start(&self, i: usize) -> usize {
        if self.diagonal {
            i + 1
        } else {
            self.c0
        }
    }

    /// Pairs contributed by each row, in row order.
    fn counts(&self) -> Vec<usize> {
        (self.r0..self.r1).map(|i| self.c1 - self.start(i)).collect()
    }
}

/// Cut `buf` into consecutive, disjoint spans of the given lengths.
///
/// The same `split_at_mut` chain `table.rs` uses: it hands each rayon task a
/// span the borrow checker already knows is exclusive, so rows write straight
/// into their final offsets instead of being collected and concatenated.
fn spans_mut<'a, T>(mut buf: &'a mut [T], counts: &[usize]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(counts.len());
    for &c in counts {
        let (head, tail) = buf.split_at_mut(c);
        out.push(head);
        buf = tail;
    }
    out
}

impl TiledReader {
    fn triangle(&self) -> bool {
        self.cols.is_none()
    }

    /// The column side: the second set, or `rows` itself for the triangle.
    fn col_side(&self) -> (&Sketches, &[i64]) {
        match &self.cols {
            Some((sketches, ids)) => (sketches, ids),
            None => (&self.rows, &self.row_ids),
        }
    }

    /// The next tile that could hold pairs, advancing the cursor past it.
    ///
    /// Row tile by row tile. The triangle starts each row of tiles on the
    /// diagonal (`tj` starts at `ti`), so tiles below it are never visited
    /// rather than being visited and discarded; the rectangle visits them all.
    fn advance(&mut self) -> Option<Tile> {
        if self.ti >= self.row_tiles || self.col_tiles == 0 {
            return None;
        }
        let (ti, tj) = (self.ti, self.tj);
        self.tj += 1;
        if self.tj >= self.col_tiles {
            self.ti += 1;
            self.tj = if self.triangle() { self.ti } else { 0 };
        }
        Some(Tile {
            r0: ti * self.side,
            r1: ((ti + 1) * self.side).min(self.n_rows),
            c0: tj * self.side,
            c1: ((tj + 1) * self.side).min(self.n_cols),
            diagonal: self.triangle() && ti == tj,
        })
    }

    fn tile_batch(&self, tile: &Tile, counts: Vec<usize>) -> Result<RecordBatch, ArrowError> {
        let total: usize = counts.iter().sum();
        let mut iv = vec![0i64; total];
        let mut jv = vec![0i64; total];
        let mut dv = vec![0f32; total];

        let si = spans_mut(&mut iv, &counts);
        let sj = spans_mut(&mut jv, &counts);
        let sd = spans_mut(&mut dv, &counts);
        let mut work: Vec<_> = si
            .into_iter()
            .zip(sj)
            .zip(sd)
            .enumerate()
            .map(|(k, ((a, b), c))| (tile.r0 + k, a, b, c))
            .collect();

        let (row_ids, weighted) = (&self.row_ids, self.weighted);
        let (cols, col_ids) = self.col_side();
        per_width_pair!(
            &*self.rows,
            cols,
            |a, b| {
                work.par_iter_mut().for_each(|(i, iv, jv, dv)| {
                    let i = *i;
                    // Indexing cannot go out of range: each side's ids were
                    // checked against its sketch count when the reader was
                    // built, and the tile is clipped to that count on each
                    // axis, so `i` and `j` are in range.
                    for (slot, j) in (tile.start(i)..tile.c1).enumerate() {
                        iv[slot] = row_ids[i];
                        jv[slot] = col_ids[j];
                        dv[slot] = unifrac_from_sketches(&a[i], &b[j], weighted);
                    }
                })
            },
            // Unreachable: two non-empty sides were refused at construction
            // unless their widths match, and an empty side yields no tile.
            // An error rather than a panic all the same, beneath an FFI
            // boundary.
            return Err(ArrowError::ComputeError(
                "rows and cols hold sketches of different widths".into()
            ))
        );

        coo_batch(&self.schema, iv, jv, dv)
    }
}

impl Iterator for TiledReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            let tile = self.advance()?;
            let counts = tile.counts();
            // Diagonal tiles of one column, and the last tile of a row when the
            // triangle ends exactly on a boundary, hold nothing. Skip them
            // rather than emitting empty batches.
            if counts.iter().all(|&c| c == 0) {
                continue;
            }
            return Some(self.tile_batch(&tile, counts).inspect_err(|_| {
                // A failed batch ends the stream: continuing would hand the
                // caller a hole in the middle of a distance matrix.
                self.done = true;
            }));
        }
    }
}

impl RecordBatchReader for TiledReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
