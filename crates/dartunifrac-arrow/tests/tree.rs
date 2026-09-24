//! What `tree_from_arrays` must reject, and — just as deliberately — what it
//! must not, because core already owns that check.

mod common;

use common::UnwrapErrNoDebug;
use dartunifrac_arrow::{tree_from_arrays, MarshalError, NO_PARENT_IN};
use dartunifrac_core::{build_sketches, CoreError, Method, SketchParams, Table, NO_PARENT};

fn assert_tree_err(e: MarshalError, needle: &str) {
    match &e {
        MarshalError::Tree(m) => assert!(
            m.contains(needle),
            "message should name the problem; got {m:?}, wanted it to contain {needle:?}"
        ),
        other => panic!("expected MarshalError::Tree, got {other:?}"),
    }
}

#[test]
fn minus_one_is_the_root_sentinel() {
    // -1 is what a C caller writes; core spells the same thing usize::MAX.
    let t = tree_from_arrays(&[NO_PARENT_IN, 0, 0], &[0.0, 1.0, 2.0]).unwrap();
    assert_eq!(t.parent, vec![NO_PARENT, 0, 0]);
    assert_eq!(t.lens, vec![0.0, 1.0, 2.0]);
}

#[test]
fn a_forest_with_several_roots_is_accepted() {
    // The binary's newick path produces one: node 0 is an unused slot with no
    // parent, alongside the real root. Rejecting multiple roots would break it.
    let t = tree_from_arrays(&[NO_PARENT_IN, NO_PARENT_IN, 1], &[0.0, 0.0, 1.0]).unwrap();
    assert_eq!(t.parent, vec![NO_PARENT, NO_PARENT, 1]);
}

#[test]
fn a_negative_parent_that_is_not_the_sentinel_is_rejected() {
    let e = tree_from_arrays(&[NO_PARENT_IN, -2], &[0.0, 1.0]).marshal_err();
    assert_tree_err(e, "-2");
}

#[test]
fn a_parent_past_the_end_of_the_tree_is_rejected() {
    // Core would catch this too, but the C API creates the tree handle in a
    // separate call from sketching: a caller should learn at df_tree_new, not
    // three calls later.
    let e = tree_from_arrays(&[NO_PARENT_IN, 7], &[0.0, 1.0]).marshal_err();
    assert_tree_err(e, "7");

    // At the boundary, where `>` and `>=` differ: node ids run 0..n, so n itself
    // is already one past the end. A `>` typo would let this through.
    let e = tree_from_arrays(&[NO_PARENT_IN, 2], &[0.0, 1.0]).marshal_err();
    assert_tree_err(e, "2");

    // And the last valid id must still be accepted.
    assert_eq!(
        tree_from_arrays(&[NO_PARENT_IN, 1, 1], &[0.0, 1.0, 1.0]).unwrap().parent,
        vec![NO_PARENT, 1, 1]
    );
}

#[test]
fn a_parent_beyond_32_bits_is_rejected_rather_than_truncated() {
    // Same hazard as sample_idx: `p as usize` on a 32-bit target would turn
    // 2^32 into 0, silently reparenting the node to the root instead of
    // rejecting it. See the i64 comparison in tree_from_arrays.
    let e = tree_from_arrays(&[NO_PARENT_IN, 1i64 << 32], &[0.0, 1.0]).marshal_err();
    assert_tree_err(e, "4294967296");
}

#[test]
fn mismatched_array_lengths_are_rejected() {
    let e = tree_from_arrays(&[NO_PARENT_IN, 0], &[0.0]).marshal_err();
    assert_tree_err(e, "2");
}

#[test]
fn a_non_finite_branch_length_is_rejected_rather_than_silently_dropped() {
    // This is the one that matters. Core selects edges with `lens[v] > 0.0`,
    // and NaN fails that comparison, so a NaN length would quietly remove an
    // edge from the id space instead of failing: a wrong answer, not an error.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let e = tree_from_arrays(&[NO_PARENT_IN, 0], &[0.0, bad]).marshal_err();
        assert_tree_err(e, "1");
    }
}

#[test]
fn a_negative_branch_length_is_accepted_because_core_already_ignores_it() {
    // Same `lens[v] > 0.0` filter, but here the silence is correct and
    // pre-existing: the binary's newick parser can produce negative lengths and
    // they have always been dropped. Rejecting them here would diverge.
    let t = tree_from_arrays(&[NO_PARENT_IN, 0], &[0.0, -1.0]).unwrap();
    assert_eq!(t.lens, vec![0.0, -1.0]);
}

#[test]
fn cycles_are_left_for_core_to_catch() {
    // Deliberate non-duplication: detecting a cycle is a traversal, core already
    // does it, and doing it twice would mean two places to keep correct. This
    // test exists so that split is a decision on the record rather than an
    // oversight -- if someone adds cycle detection here, it will fail.
    let t = tree_from_arrays(&[NO_PARENT_IN, 2, 1], &[0.0, 1.0, 1.0]).unwrap();

    let table = Table {
        n_samples: 2,
        colptr: vec![0, 1, 2],
        node: vec![1, 2],
        value: vec![],
        col_sums: vec![0.0, 0.0],
    };
    let p = SketchParams {
        k: 4,
        method: Method::Dmh,
        ers_l: 2048,
        seed: 1,
        weighted: false,
        raw_counts: false,
        portable: false,
    };
    match build_sketches(&t, &table, &p, "") {
        Err(CoreError::MalformedTable(_)) => {}
        other => panic!("core should reject the cycle, got {other:?}"),
    }
}
