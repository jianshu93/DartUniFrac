//! The sketch compatibility key, and the sha256 fingerprint that stands for it.
//!
//! Two sketch sets are comparable only when everything that shaped their values
//! matches: the tree, the sketcher and its parameters, the caller's data, the
//! hash family, this library, and the libm that tmh's `-ln(u)` draws through
//! (`localdocs/internal/id-space.md`, "The compatibility key"). Nothing about a
//! sketch shows any of it, so a mismatch gives plausible, wrong distances. The
//! fingerprint is a sha256 over all of it, which is what a stored sketch carries
//! and what an import checks.
//!
//! Every integer is hashed at a fixed width, little-endian, and never as
//! `usize`, so a wasm32 build and a 64-bit one agree on the same inputs. Each
//! string is preceded by its length, so no two keys encode to the same bytes.
//!
//! Most terms are declared: versions, a target triple. Two are measured at run
//! time, in the process that sketches. The libm probe hashes 65,536 `-ln(u)`.
//! The self-test hashes what core actually sketches on a fixed fixture, for
//! every method and weighting, so dartminhash's `mixed_tab` feature, a patched
//! or forked dartminhash, the other libm functions the sketcher calls
//! (`exp_m1`, `log2`), and a change to core between releases all change the key.
//!
//! What the key still cannot see, so that no one relies on it for it: a
//! difference that shows on none of the self-test's inputs and on too few of
//! the probe's to be drawn. A libm whose `ln` disagrees on 1 input in 10^5, or
//! a change to sketching that the fixture does not reach, can pass unseen.

use std::hint::black_box;
use std::sync::OnceLock;

use dartunifrac_core::{
    CoreError, Method, NO_PARENT, SketchParams, SketchSet, Table, Tree, build_sketches,
};
use sha2::{Digest, Sha256};

use crate::df_params_t;

/// The DartMinHash release every sketch here comes from. True only because the
/// manifest pins exactly this version; a test holds the two, and Cargo.lock, to
/// each other.
pub(crate) const DARTMINHASH_VERSION: &str = "0.1.8";

/// The key's encoding and its version. Changing how the key is encoded must
/// change this tag, which changes every fingerprint, so sketches stored under
/// the old encoding are refused rather than misread.
pub(crate) const KEY_TAG: &str = "dartunifrac/sketch-key/v1";

/// The tree digest's encoding and its version, for the same reason.
const TREE_TAG: &str = "dartunifrac/tree/v1";

/// The libm probe's encoding and its version, for the same reason.
const LIBM_TAG: &str = "dartunifrac/libm/v1";

/// The self-test's encoding, fixture and runs, and their version. Changing any
/// of them changes every fingerprint, as it must.
const SELFTEST_TAG: &str = "dartunifrac/selftest/v1";

/// How many `-ln(u)` the libm probe hashes. Two libms whose `ln` disagree on 1
/// input in 10^4 are told apart with probability 1 - (1 - 1e-4)^65536 > 0.998,
/// but at 1 in 10^5 only about half the time, and one sketch draws far more
/// than that, so a rarer disagreement can still change sketches unseen. The
/// target triple, also in the key, is the other half of the platform term.
const LIBM_PROBES: usize = 1 << 16;

/// The sha256 of a tree exactly as the caller passed it to df_tree_new: its node
/// count, every parent (-1 for a root) and every branch length's bits. That is
/// topology, branch lengths and node numbering at once, and, since the API never
/// parses Newick, it is also the "tree parser" term of the key.
pub(crate) fn tree_digest(parent: &[i64], lens: &[f64]) -> [u8; 32] {
    let mut h = Sha256::new();
    put_str(&mut h, TREE_TAG);
    h.update((parent.len() as u64).to_le_bytes());
    for &p in parent {
        h.update(p.to_le_bytes());
    }
    for &l in lens {
        h.update(l.to_bits().to_le_bytes());
    }
    h.finalize().into()
}

/// The compatibility key. Built by [`Key::new`]; a test may change any field.
#[derive(Clone, Copy)]
pub(crate) struct Key {
    pub(crate) tree: [u8; 32],
    pub(crate) method: u32,
    pub(crate) k: u32,
    pub(crate) seed: u64,
    pub(crate) bbits: u32,
    pub(crate) weighted: bool,
    pub(crate) raw_counts: bool,
    pub(crate) data_version: u64,
    pub(crate) dartminhash: &'static str,
    pub(crate) library: &'static str,
    pub(crate) target: &'static str,
    pub(crate) libm: [u8; 32],
    pub(crate) selftest: [u8; 32],
}

impl Key {
    /// The key of sketches made against a tree with this digest, under these
    /// (already validated) parameters, by this build.
    pub(crate) fn new(tree: [u8; 32], p: &df_params_t) -> Key {
        Key {
            tree,
            method: p.method,
            k: p.k,
            seed: p.seed,
            bbits: p.bbits,
            weighted: p.weighted,
            raw_counts: p.raw_counts,
            data_version: p.data_version,
            dartminhash: DARTMINHASH_VERSION,
            library: env!("CARGO_PKG_VERSION"),
            // Set by build.rs: the triple this library was compiled for.
            target: env!("DF_TARGET"),
            libm: libm_digest(),
            selftest: selftest_digest(),
        }
    }

    /// The fingerprint: sha256 over [`KEY_TAG`] and every field.
    pub(crate) fn digest(&self) -> [u8; 32] {
        self.digest_with(KEY_TAG)
    }

    /// [`Key::digest`] under another tag, so a test can show the tag is hashed.
    pub(crate) fn digest_with(&self, tag: &str) -> [u8; 32] {
        let mut h = Sha256::new();
        put_str(&mut h, tag);
        h.update(self.tree);
        h.update(self.method.to_le_bytes());
        h.update(self.k.to_le_bytes());
        h.update(self.seed.to_le_bytes());
        h.update(self.bbits.to_le_bytes());
        h.update([u8::from(self.weighted), u8::from(self.raw_counts)]);
        h.update(self.data_version.to_le_bytes());
        put_str(&mut h, self.dartminhash);
        put_str(&mut h, self.library);
        put_str(&mut h, self.target);
        h.update(self.libm);
        h.update(self.selftest);
        h.finalize().into()
    }
}

/// A string, preceded by its length so that no two sequences of strings hash
/// the same bytes.
fn put_str(h: &mut Sha256, s: &str) {
    // Lossless: usize is at most 64 bits on every target.
    h.update((s.len() as u64).to_le_bytes());
    h.update(s.as_bytes());
}

/// The six ways this API sketches: each method, unweighted, weighted, and
/// weighted on raw counts. The self-test runs every one.
pub(crate) const SELFTEST_RUNS: [(Method, bool, bool); 6] = [
    (Method::Dmh, false, false),
    (Method::Dmh, true, false),
    (Method::Dmh, true, true),
    (Method::Tmh, false, false),
    (Method::Tmh, true, false),
    (Method::Tmh, true, true),
];

/// The self-test's sketch size and seed: fixed, and small enough to cost
/// microseconds.
const SELFTEST_K: usize = 64;
const SELFTEST_SEED: u64 = 1337;

/// What this build's sketcher computes on a fixed fixture, for every method and
/// weighting: the one term of the key that is measured rather than declared. It
/// changes with anything that changes sketch values, whatever any version
/// string says: dartminhash's `mixed_tab` feature, a patched or forked
/// dartminhash, any libm function the sketcher calls (at these inputs), or a
/// change to core between releases. Computed once per process.
///
/// Call it inside the context's pool: the first call sketches, with rayon.
pub(crate) fn selftest_digest() -> [u8; 32] {
    static DIGEST: OnceLock<[u8; 32]> = OnceLock::new();
    // Not get_or_init around the sketching. A rayon thread that waits on its own
    // jobs may run another of the pool's tasks meanwhile; if that task makes a
    // fingerprint too, get_or_init would re-enter the cell on the same thread,
    // which deadlocks. Computing outside the cell costs, at worst, a second
    // identical computation.
    if let Some(d) = DIGEST.get() {
        return *d;
    }
    let d = selftest_digest_with(|tree, table, params| {
        build_sketches(tree, table, params, "(self-test) ")
    });
    *DIGEST.get_or_init(|| d)
}

/// [`selftest_digest`] through any sketcher, so a test can change one value.
/// The fixture is `data/test.nwk` with `data/test_OTU_table.txt`, the one core's
/// golden tests and this crate's tests use.
pub(crate) fn selftest_digest_with(
    run: impl Fn(&Tree, &Table, &SketchParams) -> Result<SketchSet, CoreError>,
) -> [u8; 32] {
    let tree = Tree {
        parent: vec![NO_PARENT, NO_PARENT, 1, 2, 2, 4, 4, 1, 7, 7, 9, 9, 9],
        lens: vec![
            0.0,
            0.0,
            0.30000001192092896,
            0.10000000149011612,
            0.019999999552965164,
            0.05000000074505806,
            0.05000000074505806,
            0.4000000059604645,
            0.20000000298023224,
            0.05000000074505806,
            0.10000000149011612,
            0.15000000596046448,
            0.20000000298023224,
        ],
    };
    let table = Table {
        n_samples: 3,
        colptr: vec![0, 3, 6, 9],
        node: vec![3, 6, 8, 5, 10, 11, 3, 6, 12],
        value: vec![2.0, 5.0, 3.0, 3.0, 9.0, 6.0, 4.0, 7.0, 3.0],
        col_sums: vec![10.0, 18.0, 14.0],
    };
    let mut h = Sha256::new();
    put_str(&mut h, SELFTEST_TAG);
    for (method, weighted, raw_counts) in SELFTEST_RUNS {
        let params = SketchParams {
            k: SELFTEST_K,
            method,
            ers_l: 0, // read only by ERS, which this API refuses
            seed: SELFTEST_SEED,
            weighted,
            raw_counts,
            portable: true,
        };
        // A fixed, valid fixture: failing to sketch it is a bug in this build,
        // and a panic here is a DF_ERR_PANIC at the boundary, which says so.
        let set = run(&tree, &table, &params).expect("the self-test fixture must sketch");
        h.update((set.kept.len() as u64).to_le_bytes());
        for &s in &set.kept {
            h.update((s as u64).to_le_bytes());
        }
        for sketch in &set.sketches {
            h.update((sketch.len() as u64).to_le_bytes());
            for &v in sketch {
                h.update(v.to_le_bytes());
            }
        }
    }
    h.finalize().into()
}

/// This platform's libm as tmh meets it. Computed once per process.
pub(crate) fn libm_digest() -> [u8; 32] {
    static DIGEST: OnceLock<[u8; 32]> = OnceLock::new();
    *DIGEST.get_or_init(|| libm_digest_with(f64::ln))
}

/// A sha256 over `-ln(u)` for [`LIBM_PROBES`] fixed `u` in (0, 1), drawn the way
/// dartminhash's TreeMinHash draws its exponentials (`uniform_open01`, then
/// `-u.ln()`). `ln` is not correctly rounded, and implementations differ by an
/// ulp, which changes every tmh sketch value; this is how the key sees that.
/// Generic over `ln` so a test can perturb one result.
pub(crate) fn libm_digest_with(ln: impl Fn(f64) -> f64) -> [u8; 32] {
    const DEN: f64 = (1u64 << 53) as f64;
    let mut h = Sha256::new();
    put_str(&mut h, LIBM_TAG);
    // splitmix64 from a fixed seed: the same points on every platform.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..LIBM_PROBES {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = ((z >> 11) as f64 + 0.5) / DEN;
        // black_box: with a constant input LLVM may fold ln at compile time,
        // using the build host's libm instead of the one this process runs on.
        let e = -ln(black_box(u));
        h.update(e.to_bits().to_le_bytes());
    }
    h.finalize().into()
}
