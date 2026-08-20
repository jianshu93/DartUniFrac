//! Behavioural contracts of `build_sketches`, on trees small enough to reason
//! about by hand. Where `golden.rs` pins exact values against the pre-extraction
//! engine, these pin *intent* — each one fails if the corresponding rule is
//! dropped during the extraction, even if the numbers still look plausible.

use dartunifrac_core::*;

/// Four tips under one root, every branch length 1.0:
///
/// ```text
///        0            node 0 = root (no incoming edge)
///     /  |  \  \
///    1   2   3  4     tips
/// ```
fn star_tree() -> Tree {
    Tree {
        parent: vec![NO_PARENT, 0, 0, 0, 0],
        lens: vec![0.0, 1.0, 1.0, 1.0, 1.0],
    }
}

fn params(weighted: bool, portable: bool) -> SketchParams {
    SketchParams {
        k: 64,
        method: Method::Dmh,
        ers_l: 2048,
        seed: 42,
        weighted,
        raw_counts: false,
        portable,
    }
}

/// The whole point of `--portable-sketches`: when the samples do not cover the
/// tree, the two id spaces differ, and so must the sketches. `golden.rs`'s
/// fixture cannot show this because its samples touch nearly every branch.
#[test]
fn portable_changes_sketches_when_the_samples_miss_part_of_the_tree() {
    // Samples touch tips 3 and 4; tips 1 and 2 are never reached. The run-local
    // space is then {3,4}, numbering them 0 and 1, while the portable space is
    // {1,2,3,4}, numbering the same two edges 2 and 3.
    //
    // The untouched tips must sit at *lower* node ids than the touched ones for
    // this to bite. Untouched branches numbered above every touched branch leave
    // the touched ids unshifted, so both modes agree -- which is the same reason
    // a tip appended at the end of a tree does not invalidate stored sketches.
    let table = Table {
        n_samples: 2,
        colptr: vec![0, 1, 2],
        node: vec![3, 4],
        value: vec![1.0, 1.0],
        col_sums: vec![1.0, 1.0],
    };
    let local = build_sketches(&star_tree(), &table, &params(true, false), "").unwrap();
    let portable = build_sketches(&star_tree(), &table, &params(true, true), "").unwrap();
    assert_ne!(
        local.sketches, portable.sketches,
        "portable must renumber the id space when samples miss branches"
    );
}

/// A sample's sketch under `portable` must not depend on which other samples
/// shared the call. This is the property the whole sketch-reuse design rests on.
#[test]
fn portable_sketches_do_not_depend_on_the_other_samples_in_the_call() {
    let two = Table {
        n_samples: 2,
        colptr: vec![0, 1, 2],
        node: vec![3, 4],
        value: vec![1.0, 1.0],
        col_sums: vec![1.0, 1.0],
    };
    // Same first two samples, plus a third touching tips 1 and 2 -- which sit
    // below tips 3 and 4, so they shift the run-local numbering of the first
    // two samples' edges.
    let three = Table {
        n_samples: 3,
        colptr: vec![0, 1, 2, 4],
        node: vec![3, 4, 1, 2],
        value: vec![1.0, 1.0, 1.0, 1.0],
        col_sums: vec![1.0, 1.0, 2.0],
    };
    let a = build_sketches(&star_tree(), &two, &params(true, true), "").unwrap();
    let b = build_sketches(&star_tree(), &three, &params(true, true), "").unwrap();
    assert_eq!(a.sketches[0], b.sketches[0]);
    assert_eq!(a.sketches[1], b.sketches[1]);

    // And without portable, the same comparison must fail — otherwise the test
    // above would pass for the wrong reason.
    let a = build_sketches(&star_tree(), &two, &params(true, false), "").unwrap();
    let b = build_sketches(&star_tree(), &three, &params(true, false), "").unwrap();
    assert_ne!((&a.sketches[0], &a.sketches[1]), (&b.sketches[0], &b.sketches[1]));
}

/// `col_sums` is the caller's, not something to recompute from `Table`. On a
/// sheared tree most of a sample's mass is on features that resolve to no tip,
/// so a core that re-derived the denominator would silently rescale every
/// weighted distance. Two tables identical except for `col_sums` must therefore
/// sketch differently.
#[test]
fn col_sums_are_taken_from_the_caller_not_recomputed() {
    let resolved = Table {
        n_samples: 2,
        colptr: vec![0, 2, 4],
        node: vec![1, 2, 3, 4],
        value: vec![1.0, 3.0, 2.0, 2.0],
        col_sums: vec![4.0, 4.0], // sums of the resolved entries alone
    };
    let with_unresolved = Table {
        n_samples: 2,
        colptr: vec![0, 2, 4],
        node: vec![1, 2, 3, 4],
        value: vec![1.0, 3.0, 2.0, 2.0],
        col_sums: vec![8.0, 16.0], // half / a quarter of the mass is off-tree
    };
    let a = build_sketches(&star_tree(), &resolved, &params(true, true), "").unwrap();
    let b = build_sketches(&star_tree(), &with_unresolved, &params(true, true), "").unwrap();
    assert_ne!(
        a.sketches, b.sketches,
        "col_sums must be used as the denominator, not derived from the entries"
    );
}

/// Unweighted UniFrac is set membership, so the magnitude of a count cannot
/// matter — only that the feature is present.
#[test]
fn unweighted_ignores_magnitudes() {
    let ones = Table {
        n_samples: 2,
        colptr: vec![0, 2, 4],
        node: vec![1, 2, 3, 4],
        value: vec![1.0, 1.0, 1.0, 1.0],
        col_sums: vec![2.0, 2.0],
    };
    let varied = Table {
        n_samples: 2,
        colptr: vec![0, 2, 4],
        node: vec![1, 2, 3, 4],
        value: vec![1.0, 500.0, 7.0, 3.0],
        col_sums: vec![507.0, 10.0],
    };
    let a = build_sketches(&star_tree(), &ones, &params(false, true), "").unwrap();
    let b = build_sketches(&star_tree(), &varied, &params(false, true), "").unwrap();
    assert_eq!(a.sketches, b.sketches);
}

/// A sample with no entries produces no weighted set and is dropped. The
/// surviving indices — not a shrunken count — are how the caller re-associates
/// sketches with its own sample names.
#[test]
fn empty_samples_are_dropped_and_reported_by_index() {
    let table = Table {
        n_samples: 4,
        // sample 0: tip 1 | sample 1: nothing | sample 2: tip 2 | sample 3: nothing
        colptr: vec![0, 1, 1, 2, 2],
        node: vec![1, 2],
        value: vec![1.0, 1.0],
        col_sums: vec![1.0, 0.0, 1.0, 0.0],
    };
    let got = build_sketches(&star_tree(), &table, &params(true, true), "").unwrap();
    assert_eq!(got.kept, vec![0, 2]);
    assert_eq!(got.sketches.len(), 2);
}

/// Fewer than two survivors is reported, not panicked: the CLI turns this into
/// an error message and the C API will turn it into an empty result.
#[test]
fn fewer_than_two_non_empty_samples_is_an_error_not_a_panic() {
    let table = Table {
        n_samples: 3,
        colptr: vec![0, 1, 1, 1],
        node: vec![1],
        value: vec![1.0],
        col_sums: vec![1.0, 0.0, 0.0],
    };
    assert_eq!(
        build_sketches(&star_tree(), &table, &params(true, true), ""),
        Err(CoreError::FewerThanTwoNonEmptySamples)
    );
}

/// A tree whose every branch is zero-length yields no weighted sets at all, so
/// it surfaces as "fewer than two survivors" rather than as an empty id space.
///
/// `CoreError::NoActiveEdges` is therefore **unreachable**, and it is unreachable
/// upstream too: a set entry is only ever pushed when `lens[v] > 0.0`, and the
/// empty-sample filter runs *before* the id-space check in all four of the
/// binary's builders. The variant is kept as a defensive guard mirroring those
/// `bail!`s, because the C API will let callers hand this crate arbitrary trees.
#[test]
fn a_tree_with_no_positive_edges_has_no_comparable_samples() {
    let flat = Tree {
        parent: vec![NO_PARENT, 0, 0],
        lens: vec![0.0, 0.0, 0.0],
    };
    let table = Table {
        n_samples: 2,
        colptr: vec![0, 1, 2],
        node: vec![1, 2],
        value: vec![1.0, 1.0],
        col_sums: vec![1.0, 1.0],
    };
    assert_eq!(
        build_sketches(&flat, &table, &params(true, true), ""),
        Err(CoreError::FewerThanTwoNonEmptySamples)
    );
}

/// Every sketch has length k regardless of how sparse the sample was, because
/// the pairwise stage compares them positionally.
#[test]
fn every_sketch_has_length_k() {
    let table = Table {
        n_samples: 2,
        colptr: vec![0, 1, 4],
        node: vec![1, 2, 3, 4],
        value: vec![1.0, 1.0, 1.0, 1.0],
        col_sums: vec![1.0, 3.0],
    };
    for method in [Method::Dmh, Method::Tmh, Method::Ers] {
        let p = SketchParams { method, ..params(true, true) };
        let got = build_sketches(&star_tree(), &table, &p, "").unwrap();
        for sk in &got.sketches {
            assert_eq!(sk.len(), p.k, "{method:?} returned a short sketch");
        }
    }
}

/// Malformed input is rejected rather than indexing out of bounds — the C API
/// will hand this crate whatever a caller passes across the FFI boundary.
#[test]
fn malformed_tables_are_rejected() {
    let bad_colptr = Table {
        n_samples: 2,
        colptr: vec![0, 1], // must be n_samples + 1
        node: vec![1],
        value: vec![1.0],
        col_sums: vec![1.0, 1.0],
    };
    assert!(matches!(
        build_sketches(&star_tree(), &bad_colptr, &params(true, true), ""),
        Err(CoreError::MalformedTable(_))
    ));

    let node_out_of_range = Table {
        n_samples: 2,
        colptr: vec![0, 1, 2],
        node: vec![1, 99],
        value: vec![1.0, 1.0],
        col_sums: vec![1.0, 1.0],
    };
    assert!(matches!(
        build_sketches(&star_tree(), &node_out_of_range, &params(true, true), ""),
        Err(CoreError::MalformedTable(_))
    ));
}

/// A parent pointer outside the tree must be rejected, not indexed. The climb
/// loop reads `acc[parent[v]]` directly, so this is an out-of-bounds panic
/// waiting for a caller that builds a `Tree` from something other than a parsed
/// Newick file — which is exactly what the C API will do.
#[test]
fn out_of_range_parent_pointers_are_rejected() {
    let bad = Tree {
        parent: vec![NO_PARENT, 0, 99],
        lens: vec![0.0, 1.0, 1.0],
    };
    let table = Table {
        n_samples: 2,
        colptr: vec![0, 1, 2],
        node: vec![1, 2],
        value: vec![1.0, 1.0],
        col_sums: vec![1.0, 1.0],
    };
    assert!(matches!(
        build_sketches(&bad, &table, &params(true, true), ""),
        Err(CoreError::MalformedTable(_))
    ));
}

/// A cycle in the parent pointers must be rejected rather than hung on.
///
/// The unweighted climb happens to self-terminate, because it stops at the first
/// already-marked node; the weighted climb has no such guard and would spin
/// forever. Both are checked so the guard cannot be removed on the grounds that
/// "one of them is fine".
#[test]
fn cyclic_parent_pointers_are_rejected_rather_than_hung_on() {
    // 3 -> 4 -> 3, with a legitimate root alongside.
    let cyclic = Tree {
        parent: vec![NO_PARENT, 0, 0, 4, 3],
        lens: vec![0.0, 1.0, 1.0, 1.0, 1.0],
    };
    let table = Table {
        n_samples: 2,
        colptr: vec![0, 1, 2],
        node: vec![1, 2],
        value: vec![1.0, 1.0],
        col_sums: vec![1.0, 1.0],
    };
    for weighted in [true, false] {
        assert!(
            matches!(
                build_sketches(&cyclic, &table, &params(weighted, true), ""),
                Err(CoreError::MalformedTable(_))
            ),
            "cycle not rejected with weighted={weighted}"
        );
    }
}

/// A tree that is a legitimate forest — several roots, no cycle — must still be
/// accepted, so the cycle check cannot be tightened into rejecting valid input.
#[test]
fn multi_root_forests_are_accepted() {
    let forest = Tree {
        parent: vec![NO_PARENT, 0, NO_PARENT, 2, 2],
        lens: vec![0.0, 1.0, 0.0, 1.0, 1.0],
    };
    let table = Table {
        n_samples: 2,
        colptr: vec![0, 1, 3],
        node: vec![1, 3, 4],
        value: vec![1.0, 1.0, 1.0],
        col_sums: vec![1.0, 2.0],
    };
    assert!(build_sketches(&forest, &table, &params(true, true), "").is_ok());
}

/// `parent` and `lens` disagreeing on length would make every `lens[v]` lookup a
/// potential panic.
#[test]
fn mismatched_tree_array_lengths_are_rejected() {
    let bad = Tree {
        parent: vec![NO_PARENT, 0, 0],
        lens: vec![0.0, 1.0],
    };
    let table = Table {
        n_samples: 2,
        colptr: vec![0, 1, 2],
        node: vec![1, 2],
        value: vec![1.0, 1.0],
        col_sums: vec![1.0, 1.0],
    };
    assert!(matches!(
        build_sketches(&bad, &table, &params(true, true), ""),
        Err(CoreError::MalformedTable(_))
    ));
}

/// Presence mode never reads a weight, so it must not require the caller to
/// invent one. Before this was allowed, the binary's unweighted BIOM path
/// synthesized an nnz-sized array of `1.0` purely to satisfy the type, and then
/// permuted and copied it twice — pure waste at a scale where the table is the
/// dominant allocation.
///
/// The contract is the strong form: an empty `value` must give **bit-identical**
/// sketches to a `value` full of ones, not merely similar ones. Anything weaker
/// would let presence mode quietly depend on magnitudes.
#[test]
fn presence_mode_accepts_an_empty_value_array_and_ignores_values_entirely() {
    let tree = star_tree();
    let colptr = vec![0, 2, 4];
    let node = vec![1, 2, 3, 4];
    let with_ones = Table {
        n_samples: 2,
        colptr: colptr.clone(),
        node: node.clone(),
        value: vec![1.0, 1.0, 1.0, 1.0],
        col_sums: vec![0.0, 0.0],
    };
    let without = Table {
        n_samples: 2,
        colptr: colptr.clone(),
        node: node.clone(),
        value: Vec::new(),
        col_sums: vec![0.0, 0.0],
    };
    // And a third with wildly different magnitudes, to show presence mode is not
    // merely tolerating the values but ignoring them.
    let with_junk = Table {
        n_samples: 2,
        colptr,
        node,
        value: vec![1e9, 7.5, -3.0, 0.25],
        col_sums: vec![0.0, 0.0],
    };
    let p = params(false, false);
    let a = build_sketches(&tree, &with_ones, &p, "").expect("ones");
    let b = build_sketches(&tree, &without, &p, "").expect("empty value");
    let c = build_sketches(&tree, &with_junk, &p, "").expect("junk values");
    assert_eq!(a, b, "presence mode must not depend on value[] being present");
    assert_eq!(a, c, "presence mode must not depend on value[] contents");
}

/// The other half of the contract: weighted mode *does* read every value, so an
/// empty `value` there is a caller bug and must be reported as one. Indexing
/// `value[kk]` would otherwise panic inside a rayon worker, which across an FFI
/// boundary means aborting the host process rather than returning an error.
#[test]
fn weighted_mode_rejects_an_empty_value_array_rather_than_panicking() {
    let tree = star_tree();
    let table = Table {
        n_samples: 2,
        colptr: vec![0, 2, 4],
        node: vec![1, 2, 3, 4],
        value: Vec::new(),
        col_sums: vec![2.0, 2.0],
    };
    match build_sketches(&tree, &table, &params(true, false), "") {
        Err(CoreError::MalformedTable(m)) => {
            assert!(
                m.contains("value"),
                "the message should name the offending field, got: {m}"
            );
        }
        other => panic!("expected MalformedTable, got {other:?}"),
    }
}

/// Allowing an empty `value` must not weaken the length check for a non-empty
/// one: a short value array is still a caller bug, not a licence to read past
/// the end.
#[test]
fn a_value_array_that_is_neither_empty_nor_full_length_is_rejected() {
    let tree = star_tree();
    let table = Table {
        n_samples: 2,
        colptr: vec![0, 2, 4],
        node: vec![1, 2, 3, 4],
        value: vec![1.0, 1.0, 1.0], // one short
        col_sums: vec![0.0, 0.0],
    };
    assert!(matches!(
        build_sketches(&tree, &table, &params(false, false), ""),
        Err(CoreError::MalformedTable(_))
    ));
}
