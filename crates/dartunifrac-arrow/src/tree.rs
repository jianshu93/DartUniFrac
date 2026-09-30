use dartunifrac_core::Tree;

use crate::MarshalError;

/// Sentinel the C API uses in `parent[]` for a node with no parent.
///
/// `dartunifrac_core::NO_PARENT` is `usize::MAX`, which has no `i64` spelling;
/// `-1` is the convention a C caller already expects.
pub const NO_PARENT_IN: i64 = -1;

/// Convert a caller's `parent[]` / `branch_length[]` into a [`Tree`].
///
/// Rejects length mismatches, parents outside `0..n` other than
/// [`NO_PARENT_IN`], and non-finite branch lengths. **Cycles are not detected
/// here** — `dartunifrac_core::build_sketches` already does that, and
/// duplicating a graph traversal to say the same thing twice is not worth it.
/// Negative lengths are accepted for the same reason: core's `lens[v] > 0.0`
/// edge filter already ignores them, and the binary's newick path can produce
/// them.
pub fn tree_from_arrays(parent: &[i64], branch_length: &[f64]) -> Result<Tree, MarshalError> {
    if parent.len() != branch_length.len() {
        return Err(MarshalError::Tree(format!(
            "parent has {} entries but branch_length has {}; both are indexed by node id",
            parent.len(),
            branch_length.len()
        )));
    }
    let n = parent.len();

    let mut out = Vec::with_capacity(n);
    for (v, &p) in parent.iter().enumerate() {
        out.push(if p == NO_PARENT_IN {
            dartunifrac_core::NO_PARENT
        } else if p < 0 {
            return Err(MarshalError::Tree(format!(
                "node {v}: parent {p} is negative but is not {NO_PARENT_IN}, the root sentinel"
            )));
        } else if p >= n as i64 {
            // i64 comparison, not `p as usize >= n`: on a 32-bit target the cast
            // truncates first and a parent of 2^32 would read as 0 -- a silently
            // rewired tree rather than a rejected one.
            return Err(MarshalError::Tree(format!(
                "node {v}: parent {p} is outside the tree, which has {n} nodes"
            )));
        } else {
            p as usize
        });
    }

    for (v, &l) in branch_length.iter().enumerate() {
        if !l.is_finite() {
            return Err(MarshalError::Tree(format!(
                "node {v}: branch length {l} is not finite. Core selects edges with \
                 `len > 0.0`, a comparison every non-finite value fails, so accepting \
                 this would drop the edge from the id space instead of reporting it"
            )));
        }
    }

    Ok(Tree { parent: out, lens: branch_length.to_vec() })
}
