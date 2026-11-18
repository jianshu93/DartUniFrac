//! Golden vectors captured from the pre-extraction engine.
//!
//! These pin `build_sketches` to the exact output the `dartunifrac` binary
//! produced before the engine was moved out of it, so the extraction cannot
//! quietly change results. The fixture is the bundled `data/test.nwk` (7 tips)
//! with `data/test_OTU_table.txt` (3 samples), transcribed into code because
//! this crate does not read files.
//!
//! Captured at commit b367a5f by calling `build_sketches_simple` and
//! `build_sketches_weighted_simple` directly with k=16, seed=1337, method dmh.
//!
//! **Do not edit an expected value here.** If one of these fails, the
//! extraction changed behaviour — that is the bug, not the fixture.
//!
//! The one exception: the two ERS vectors were recaptured from `build_sketches`
//! when dartminhash moved from 0.1.8 to 0.1.9. That release rewrote ERS: it
//! densifies by Algorithm 3, and samples by binary search where 0.1.8 used an alias
//! table, which its author found sometimes fails to estimate. So every ERS sketch
//! changed. The DMH vectors and the TMH test did not, and pass unchanged. From here
//! on the ERS vectors guard 0.1.9's output against drift only. The DMH vectors
//! still verify the extraction; the ERS ones did so at 0.1.8.

use dartunifrac_core::*;

/// `data/test.nwk` as `build_parent_and_lens_simple` numbers it.
///
/// Node 0 is the `newick` crate's unused slot: no parent, no length, no edge.
/// It is retained because `total` is `max_id + 1`, and dropping it would
/// renumber every other node and therefore change every sketch.
fn test_tree() -> Tree {
    Tree {
        parent: vec![NO_PARENT, NO_PARENT, 1, 2, 2, 4, 4, 1, 7, 7, 9, 9, 9],
        lens: vec![0.0, 0.0, 0.30000001192092896, 0.10000000149011612, 0.019999999552965164, 0.05000000074505806, 0.05000000074505806, 0.4000000059604645, 0.20000000298023224, 0.05000000074505806, 0.10000000149011612, 0.15000000596046448, 0.20000000298023224],
    }
}

/// `data/test_OTU_table.txt` resolved to node ids, in the order the dense TSV
/// reader walks it: ascending feature row within each sample, non-positive
/// values already dropped.
fn test_table() -> Table {
    Table {
        n_samples: 3,
        colptr: vec![0, 3, 6, 9],
        node: vec![3, 6, 8, 5, 10, 11, 3, 6, 12],
        value: vec![2.0, 5.0, 3.0, 3.0, 9.0, 6.0, 4.0, 7.0, 3.0],
        col_sums: vec![10.0, 18.0, 14.0],
    }
}

fn params(weighted: bool, portable: bool) -> SketchParams {
    SketchParams {
        k: 16,
        method: Method::Dmh,
        ers_l: 2048,
        seed: 1337,
        weighted,
        raw_counts: false,
        portable,
    }
}

fn weighted_expected() -> Vec<Vec<u64>> {
    vec![
    vec![
        6109215235811546771, 4448982827566103532, 12110907102781106075, 12294545009166051657,
        15426295578266653425, 3334286281215730996, 15768018314345371448, 5720882680924996318,
        2511528848265521286, 6547995086521894532, 10028829507641750804, 17753617789732528999,
        14859319188605807875, 2002964490006572299, 8551988683501910054, 1070276508455420232,
    ],
    vec![
        15531856672663274222, 4262156677074979330, 2681316882404831282, 1021665016608968308,
        447691480975264033, 3334286281215730996, 4268717054684466627, 5720882680924996318,
        2511528848265521286, 1840514995738727734, 11783727686631769286, 17753617789732528999,
        7654744927455452080, 17574157738776293419, 8551988683501910054, 7354337112599138843,
    ],
    vec![
        6109215235811546771, 2526377721590892059, 12110907102781106075, 12294545009166051657,
        6093471332401294231, 3334286281215730996, 15768018314345371448, 8678925695399213401,
        10567092133569749890, 12535352795753059544, 10028829507641750804, 17753617789732528999,
        14859319188605807875, 17574157738776293419, 8551988683501910054, 1070276508455420232,
    ],
    ]
}

fn unweighted_expected() -> Vec<Vec<u64>> {
    vec![
    vec![
        17065539452326189473, 5388483271961042793, 2681316882404831282, 4590298958847932483,
        447691480975264033, 3737486821560176218, 7292040649426970002, 7107655196361497490,
        11485139032177109881, 12145355677831972818, 17933540549013582117, 17753617789732528999,
        14859319188605807875, 7057098328671626849, 10840000511321464814, 14935665196011265833,
    ],
    vec![
        17065539452326189473, 4262156677074979330, 2681316882404831282, 9573224903840904532,
        447691480975264033, 3334286281215730996, 7292040649426970002, 5720882680924996318,
        10567092133569749890, 8208146992953324030, 11783727686631769286, 17753617789732528999,
        7654744927455452080, 7057098328671626849, 10840000511321464814, 14935665196011265833,
    ],
    vec![
        9876401559875971721, 5388483271961042793, 2681316882404831282, 4590298958847932483,
        447691480975264033, 3334286281215730996, 7292040649426970002, 5720882680924996318,
        10567092133569749890, 12535352795753059544, 17933540549013582117, 17753617789732528999,
        14859319188605807875, 14728728787860126126, 10840000511321464814, 12003129962093404539,
    ],
    ]
}

#[test]
fn weighted_sketches_match_the_pre_extraction_engine() {
    let got = build_sketches(&test_tree(), &test_table(), &params(true, false), "")
        .expect("all three samples are non-empty");
    assert_eq!(got.kept, vec![0, 1, 2]);
    assert_eq!(got.sketches, weighted_expected());
}

#[test]
fn unweighted_sketches_match_the_pre_extraction_engine() {
    let got = build_sketches(&test_tree(), &test_table(), &params(false, false), "")
        .expect("all three samples are non-empty");
    assert_eq!(got.kept, vec![0, 1, 2]);
    assert_eq!(got.sketches, unweighted_expected());
}

/// Every sample in this fixture touches nearly the whole tree and k=16 far
/// exceeds the edge count, so the two id spaces happen to agree here. That is a
/// property of the fixture, not of the flag — `portable_changes_sketches_when_`
/// `the_samples_miss_part_of_the_tree` in the id-space suite is what proves the
/// flag does anything. Pinned so the agreement stays deliberate.
#[test]
fn portable_agrees_with_run_local_on_this_fixture() {
    let d = build_sketches(&test_tree(), &test_table(), &params(true, false), "").unwrap();
    let p = build_sketches(&test_tree(), &test_table(), &params(true, true), "").unwrap();
    assert_eq!(d.sketches, p.sketches);
}

// ---------------------------------------------------------------------------
// The remaining (method x raw_counts) combinations, same fixture and seed.
//
// Without these, only Dmh + relative abundance was pinned, and the ERS cap
// construction or the raw-counts branch could have changed unnoticed.
// ---------------------------------------------------------------------------

/// Golden: weighted mode with --raw-counts: counts go into the accumulation unnormalized.
#[test]
fn weighted_dmh_raw_counts_matches_the_pre_extraction_engine() {
    let p = SketchParams {
        k: 16,
        method: Method::Dmh,
        ers_l: 2048,
        seed: 1337,
        weighted: true,
        raw_counts: true,
        portable: false,
    };
    let expected: Vec<Vec<u64>> = vec![
        vec![
            10262813384726359226, 5413229176952027858, 18366968479918701358, 3719030629323357682,
            4066451085736158922, 13656116162742777531, 7292040649426970002, 7107655196361497490,
            11485139032177109881, 1616395416016452964, 4507231795959348086, 4729998807800222471,
            10080522022619715810, 7057098328671626849, 10840000511321464814, 5245805945312911441,
        ],
        vec![
            4490697366102493764, 5402224597387870055, 11009166144055423128, 14541688019284851988,
            136982447369458862, 15535703384869716103, 8627308336958368302, 5619996299406521099,
            2150524268635233835, 17745689220774140065, 5049275646796919411, 2500610208625924222,
            808621945466142465, 1585447237426221037, 10840000511321464814, 5245805945312911441,
        ],
        vec![
            9876401559875971721, 5413229176952027858, 17409307539898182186, 7246943193272678113,
            4066451085736158922, 9594965853229343256, 7292040649426970002, 3777135584277498452,
            2150524268635233835, 1616395416016452964, 4507231795959348086, 4729998807800222471,
            10080522022619715810, 14728728787860126126, 10840000511321464814, 9798732015522930618,
        ],
    ];
    let got = build_sketches(&test_tree(), &test_table(), &p, "").unwrap();
    assert_eq!(got.sketches, expected);
}

/// Golden: weighted Efficient Rejection Sampling, whose caps are the run's max weights.
#[test]
fn weighted_ers_matches_the_pre_extraction_engine() {
    let p = SketchParams {
        k: 16,
        method: Method::Ers,
        ers_l: 2048,
        seed: 1337,
        weighted: true,
        raw_counts: false,
        portable: false,
    };
    let expected: Vec<Vec<u64>> = vec![
        vec![
            6763249926781496728, 17623970487765242034, 16943365417359461439, 8709703885886891809,
            3150063961872558098, 8857813616877978796, 17181043418368412562, 59619151913402147,
            17000420279537294063, 17894083459557123984, 5624136161549993714, 5055538797895281431,
            5906705630757318325, 5169693397591530899, 11500939146571854573, 16243033676482171293,
        ],
        vec![
            6763249926781496728, 14244469173574963132, 16943365417359461439, 12176316814105784227,
            15560616502081529318, 13927209537354439043, 16482015281853929148, 59619151913402147,
            7701561557143421077, 17894083459557123984, 239170630496211001, 5635385426406648826,
            15678613800291742165, 2806795484297779791, 12135814800428041683, 13744125782542977146,
        ],
        vec![
            2877310282941382090, 9277969553372495767, 16943365417359461439, 8709703885886891809,
            3150063961872558098, 8857813616877978796, 17181043418368412562, 59619151913402147,
            17000420279537294063, 17894083459557123984, 5624136161549993714, 5635385426406648826,
            11134195609854011608, 5169693397591530899, 11500939146571854573, 16243033676482171293,
        ],
    ];
    let got = build_sketches(&test_tree(), &test_table(), &p, "").unwrap();
    assert_eq!(got.sketches, expected);
}

/// Golden: unweighted ERS, whose caps come from branch lengths.
#[test]
fn unweighted_ers_matches_the_pre_extraction_engine() {
    let p = SketchParams {
        k: 16,
        method: Method::Ers,
        ers_l: 2048,
        seed: 1337,
        weighted: false,
        raw_counts: false,
        portable: false,
    };
    let expected: Vec<Vec<u64>> = vec![
        vec![
            1176930476455241774, 12763888015962036405, 15580323643041902896, 11378849182141322725,
            16273920274197286603, 9348493655519226120, 8355321322717475546, 17646889577669206454,
            13631067010954416638, 14736753616577617485, 9532830286090903658, 14510862856184737609,
            4147084380684332076, 10286662332790093061, 3953649375215645483, 13873213702989914424,
        ],
        vec![
            1176930476455241774, 1799473158321581741, 15580323643041902896, 5883623558701306646,
            9463491886192863723, 9348493655519226120, 8355321322717475546, 17646889577669206454,
            10649837040572292597, 14736753616577617485, 8946475685710517441, 5006858515869642176,
            4147084380684332076, 9726553761866279128, 18066284977301619380, 2149361784095787775,
        ],
        vec![
            1176930476455241774, 1799473158321581741, 15580323643041902896, 6252031340681417278,
            6397274762031156181, 9348493655519226120, 8355321322717475546, 17646889577669206454,
            13631067010954416638, 14736753616577617485, 9532830286090903658, 14510862856184737609,
            9963945902168210396, 10286662332790093061, 13327077594018219458, 13873213702989914424,
        ],
    ];
    let got = build_sketches(&test_tree(), &test_table(), &p, "").unwrap();
    assert_eq!(got.sketches, expected);
}

/// TreeMinHash's exact output cannot be pinned portably, so this pins everything
/// around it instead.
///
/// `tmh` draws an exponential variate per element as `-ln(u)`
/// (`dartminhash::treeminhash`), and `ln` is libm: not correctly rounded, and free
/// to differ by an ulp between platforms. Perturbing that draw by exactly one ulp
/// was measured to change every `tmh` sketch value on this fixture while leaving
/// all of the `dmh` and `ers` vectors above untouched — `dmh` uses `ln` only for
/// an integer count and `ers` uses no transcendentals at all. An earlier version
/// of this file asserted fixed `tmh` vectors and duly failed on a maintainer's
/// machine while passing on the one that generated them.
///
/// That is a property of the sketcher, not of this crate. What the extraction
/// could break for `tmh` is every stage *before* the sketcher — validation, the
/// leaf-to-root accumulation, the empty-sample drop, id compaction — and
/// `Method` is only consulted in the final `sketch_all` call, so all of it is
/// shared with `dmh` and `ers` and already pinned exactly above.
///
/// So this asserts the `tmh`-specific properties that hold on any libm. The first
/// is the load-bearing one: on this fixture the run-local and portable id spaces
/// are the same set (its active edge count is identical either way), so the two
/// must produce *identical* sketches. That is an exact equality check on id
/// compaction which compares two runs on the same libm, and it fails if
/// compaction is broken for `tmh`.
#[test]
fn tmh_is_internally_consistent_where_exact_values_cannot_be_portable() {
    for weighted in [false, true] {
        let mut p = SketchParams {
            k: 16,
            method: Method::Tmh,
            ers_l: 2048,
            seed: 1337,
            weighted,
            raw_counts: false,
            portable: false,
        };
        let run_local = build_sketches(&test_tree(), &test_table(), &p, "").unwrap();

        assert_eq!(run_local.kept, vec![0, 1, 2], "weighted={weighted}: every sample survives");
        assert_eq!(run_local.sketches.len(), 3);
        for s in &run_local.sketches {
            assert_eq!(s.len(), p.k, "weighted={weighted}: every sketch has length k");
        }

        // Same call twice: nothing may depend on iteration or thread order.
        let again = build_sketches(&test_tree(), &test_table(), &p, "").unwrap();
        assert_eq!(run_local, again, "weighted={weighted}: sketching must be deterministic");

        // The two id spaces coincide on this fixture, so the sketches must too.
        p.portable = true;
        let portable = build_sketches(&test_tree(), &test_table(), &p, "").unwrap();
        assert_eq!(
            run_local, portable,
            "weighted={weighted}: this fixture touches every positive-length edge, so the \
             run-local and portable id spaces are the same set and must sketch identically"
        );
    }
}
