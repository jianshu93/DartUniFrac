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

impl Sketches {
    /// How many samples, i.e. how many rows the triangle has.
    fn count(&self) -> usize {
        per_width!(self, |rows| rows.len())
    }

    /// Reject what the kernel would otherwise panic on or quietly turn to NaN.
    ///
    /// Both checks exist because this crate sits beneath an FFI boundary.
    /// `count_mismatches` asserts equal lengths, and an assert there aborts the
    /// host process; a zero-length sketch divides by zero and yields a complete
    /// distance matrix that is silently meaningless.
    fn validate(&self) -> Result<(), MarshalError> {
        per_width!(self, |rows| {
            let Some(first) = rows.first() else {
                return Ok(());
            };
            let k = first.len();
            // Only an error where it would produce a value: with fewer than two
            // samples there are no pairs, so there is nothing to be wrong.
            if k == 0 && rows.len() > 1 {
                return Err(MarshalError::Sketch(
                    "sketches hold no values, so every distance would be NaN".into(),
                ));
            }
            for (s, row) in rows.iter().enumerate() {
                if row.len() != k {
                    return Err(MarshalError::Sketch(format!(
                        "sketch {s} holds {} values but sketch 0 holds {k}; \
                         sketches of different lengths are not comparable",
                        row.len()
                    )));
                }
            }
            Ok(())
        })
    }
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
/// offsets, so the stream is identical for any thread count and any
/// `batch_rows`. Only the batch boundaries move.
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
pub fn distances_reader(
    sketches: Sketches,
    ids: Vec<i64>,
    weighted: bool,
    batch_rows: usize,
) -> Result<Box<dyn RecordBatchReader + Send>, MarshalError> {
    sketches.validate()?;
    let n = sketches.count();
    if ids.len() != n {
        return Err(MarshalError::Sketch(format!(
            "{} ids were given for {n} sketches; they index the same samples",
            ids.len()
        )));
    }
    // Load-bearing, not cosmetic: `n.div_ceil(0)` below panics, and a panic
    // beneath an FFI boundary aborts the host process.
    let side = batch_rows.isqrt().max(1);
    Ok(Box::new(TiledReader {
        sketches,
        ids,
        weighted,
        n,
        side,
        tiles: n.div_ceil(side),
        ti: 0,
        tj: 0,
        schema: Arc::new(output_schema()),
        done: false,
    }))
}

struct TiledReader {
    sketches: Sketches,
    ids: Vec<i64>,
    weighted: bool,
    n: usize,
    side: usize,
    tiles: usize,
    ti: usize,
    tj: usize,
    schema: SchemaRef,
    done: bool,
}

/// One tile of the triangle: rows `r0..r1` against columns `c0..c1`.
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
    /// The next tile that could hold pairs, advancing the cursor past it.
    ///
    /// Walks the upper block triangle only: `tj` starts at `ti`, so tiles below
    /// the diagonal are never visited rather than being visited and discarded.
    fn advance(&mut self) -> Option<Tile> {
        if self.ti >= self.tiles {
            return None;
        }
        let (ti, tj) = (self.ti, self.tj);
        self.tj += 1;
        if self.tj >= self.tiles {
            self.ti += 1;
            self.tj = self.ti;
        }
        Some(Tile {
            r0: ti * self.side,
            r1: ((ti + 1) * self.side).min(self.n),
            c0: tj * self.side,
            c1: ((tj + 1) * self.side).min(self.n),
            diagonal: ti == tj,
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

        let (ids, weighted) = (&self.ids, self.weighted);
        per_width!(&self.sketches, |rows| {
            work.par_iter_mut().for_each(|(i, iv, jv, dv)| {
                let i = *i;
                // `ids[..]` and `rows[..]` are unchecked because both were
                // sized against `n` in `distances_reader`, and the tile is
                // clipped to `n` on both axes, so `i` and `j` are in range.
                for (slot, j) in (tile.start(i)..tile.c1).enumerate() {
                    iv[slot] = ids[i];
                    jv[slot] = ids[j];
                    dv[slot] = unifrac_from_sketches(&rows[i], &rows[j], weighted);
                }
            })
        });

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
