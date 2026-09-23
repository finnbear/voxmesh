//! Stairs: a half slab with a quarter step on it, in any orientation.
//!
//! Each face of a stair is what its two boxes together present to it —
//! one full quad, two at different depths, or an L — and a neighbor hides
//! exactly the part of that it actually covers.

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod common;

use common::*;
use glam::Vec3;
use voxmesh::*;

/// Every quad on `face`, as its vertices.
fn quads_on(q: &Quads, face: AlignedFace, block: TestBlock) -> Vec<[Vec3; 4]> {
    q.faces[face.index()]
        .iter()
        .map(|quad| quad.positions(face, block.shape()))
        .collect()
}

/// The `(min, max)` corner of a quad.
fn extent(quad: &[Vec3; 4]) -> (Vec3, Vec3) {
    quad.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), v| (lo.min(*v), hi.max(*v)),
    )
}

/// Asserts that `face` of a lone `block` shows exactly the boxes in
/// `want`, each given as `(min, max)`, in any order.
fn assert_face_is(block: TestBlock, face: AlignedFace, want: &[([f32; 3], [f32; 3])]) {
    let q = mesh_single(block);
    let mut got: Vec<_> = quads_on(&q, face, block)
        .iter()
        .map(|quad| {
            let (lo, hi) = extent(quad);
            (lo.to_array(), hi.to_array())
        })
        .collect();
    let key = |e: &([f32; 3], [f32; 3])| {
        let r = |v: f32| (v * 16.0).round() as i32;
        (
            [r(e.0[0]), r(e.0[1]), r(e.0[2])],
            [r(e.1[0]), r(e.1[1]), r(e.1[2])],
        )
    };
    got.sort_by_key(key);
    let mut want: Vec<_> = want.to_vec();
    want.sort_by_key(key);
    let got_keys: Vec<_> = got.iter().map(key).collect();
    let want_keys: Vec<_> = want.iter().map(key).collect();
    assert_eq!(got_keys, want_keys, "{block:?} {face:?}");
}

#[test]
fn a_floor_stair_has_ten_quads() {
    assert_eq!(mesh_single(TestBlock::Stair).total(), 10);
}

#[test]
fn a_lone_block_meshes_like_a_lone_cell() {
    for block in [TestBlock::Stair, TestBlock::Corbel, TestBlock::WallStair] {
        let from_chunk = mesh_single(block);
        let from_block = mesh_block(&block, ());
        assert_eq!(from_chunk.total(), from_block.total(), "{block:?}");
        for face in AlignedFace::ALL {
            assert_eq!(
                from_chunk.faces[face.index()].len(),
                from_block.faces[face.index()].len(),
                "{block:?} {face:?}"
            );
        }
    }
}

#[test]
fn a_floor_stair_shows_its_profile() {
    use TestBlock::Stair;
    // The slab fills the floor; the slab and the step fill the back.
    assert_face_is(
        Stair,
        AlignedFace::NegY,
        &[([0.0, 0.0, 0.0], [1.0, 0.0, 1.0])],
    );
    assert_face_is(
        Stair,
        AlignedFace::NegZ,
        &[([0.0, 0.0, 0.0], [1.0, 1.0, 0.0])],
    );
    // On top: the slab's exposed half at 0.5 and the step's top at 1.
    assert_face_is(
        Stair,
        AlignedFace::PosY,
        &[
            ([0.0, 0.5, 0.5], [1.0, 0.5, 1.0]),
            ([0.0, 1.0, 0.0], [1.0, 1.0, 0.5]),
        ],
    );
    // In front: the slab's face at the boundary and the riser halfway in.
    assert_face_is(
        Stair,
        AlignedFace::PosZ,
        &[
            ([0.0, 0.0, 1.0], [1.0, 0.5, 1.0]),
            ([0.0, 0.5, 0.5], [1.0, 1.0, 0.5]),
        ],
    );
    // Each side: the L.
    for face in [AlignedFace::PosX, AlignedFace::NegX] {
        let x = if face.is_positive() { 1.0 } else { 0.0 };
        assert_face_is(
            Stair,
            face,
            &[
                ([x, 0.0, 0.0], [x, 0.5, 1.0]),
                ([x, 0.5, 0.0], [x, 1.0, 0.5]),
            ],
        );
    }
}

#[test]
fn a_corbel_is_the_stair_upside_down() {
    use TestBlock::Corbel;
    assert_eq!(mesh_single(Corbel).total(), 10);
    assert_face_is(
        Corbel,
        AlignedFace::PosY,
        &[([0.0, 1.0, 0.0], [1.0, 1.0, 1.0])],
    );
    assert_face_is(
        Corbel,
        AlignedFace::NegY,
        &[
            ([0.0, 0.5, 0.5], [1.0, 0.5, 1.0]),
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.5]),
        ],
    );
    assert_face_is(
        Corbel,
        AlignedFace::PosZ,
        &[
            ([0.0, 0.5, 1.0], [1.0, 1.0, 1.0]),
            ([0.0, 0.0, 0.5], [1.0, 0.5, 0.5]),
        ],
    );
}

#[test]
fn a_wall_stair_stands_on_its_side() {
    use TestBlock::WallStair;
    assert_eq!(mesh_single(WallStair).total(), 10);
    // Floor on the +X wall, back on the floor.
    assert_face_is(
        WallStair,
        AlignedFace::PosX,
        &[([1.0, 0.0, 0.0], [1.0, 1.0, 1.0])],
    );
    assert_face_is(
        WallStair,
        AlignedFace::NegY,
        &[([0.0, 0.0, 0.0], [1.0, 0.0, 1.0])],
    );
    // Opposite the floor: the slab's inner face high up, the step's outer
    // face low down.
    assert_face_is(
        WallStair,
        AlignedFace::NegX,
        &[
            ([0.5, 0.5, 0.0], [0.5, 1.0, 1.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.5, 1.0]),
        ],
    );
    // The sides run along Z and show the L.
    assert_face_is(
        WallStair,
        AlignedFace::PosZ,
        &[
            ([0.5, 0.0, 1.0], [1.0, 1.0, 1.0]),
            ([0.0, 0.0, 1.0], [0.5, 0.5, 1.0]),
        ],
    );
}

#[test]
fn a_block_on_the_step_hides_only_the_step_top() {
    let q = mesh_with(&[(0, 0, 0, TestBlock::Stair), (0, 1, 0, TestBlock::Stone)]);
    // The stone hides the step's top and nothing else; the slab's inset
    // top is still open. The stair's top covers only half the plane, so
    // the stone's underside stays.
    assert_eq!(q.total(), 10 - 1 + 6);
    let tops = quads_on(&q, AlignedFace::PosY, TestBlock::Stair);
    assert!(
        tops.iter()
            .any(|quad| (extent(quad).0.y - 0.5).abs() < 1e-6),
        "the slab's exposed top is missing"
    );
}

#[test]
fn a_block_behind_the_back_hides_both_faces() {
    let q = mesh_with(&[(0, 0, 1, TestBlock::Stair), (0, 0, 0, TestBlock::Stone)]);
    // The back is full, so the two faces on the boundary hide each other.
    assert_eq!(q.total(), 10 - 1 + 6 - 1);
}

#[test]
fn a_lower_slab_beside_the_stair_hides_the_slab_half_of_the_l() {
    let q = mesh_with(&[(0, 0, 0, TestBlock::Stair), (1, 0, 0, TestBlock::LowerSlab)]);
    // The slab covers the lower strip of the stair's +X side and the
    // stair covers the whole of the slab's -X face; the step's quarter
    // stays.
    assert_eq!(q.total(), 10 - 1 + 6 - 1);
    let sides = quads_on(&q, AlignedFace::PosX, TestBlock::Stair);
    let stair_sides: Vec<_> = sides
        .iter()
        .filter(|quad| (extent(quad).0.x - 1.0).abs() < 1e-6)
        .collect();
    assert_eq!(stair_sides.len(), 1);
    assert!(
        (extent(stair_sides[0]).0.y - 0.5).abs() < 1e-6,
        "the step stays"
    );
}

#[test]
fn two_stairs_in_a_row_hide_their_shared_side_and_merge_the_rest() {
    let q = mesh_with(&[(0, 0, 0, TestBlock::Stair), (1, 0, 0, TestBlock::Stair)]);
    // Sides between them hidden in full (the L matches the L). The floor
    // and the back merge into one quad each; the two-quad faces merge
    // along X, which each part spans in full; the outer sides stay as two
    // Ls.
    assert_eq!(face_count(&q, AlignedFace::NegY), 1);
    assert_eq!(face_count(&q, AlignedFace::NegZ), 1);
    assert_eq!(face_count(&q, AlignedFace::PosY), 2);
    assert_eq!(face_count(&q, AlignedFace::PosZ), 2);
    assert_eq!(face_count(&q, AlignedFace::PosX), 2);
    assert_eq!(face_count(&q, AlignedFace::NegX), 2);
    let step_tops = quads_on(&q, AlignedFace::PosY, TestBlock::Stair);
    assert!(
        step_tops
            .iter()
            .all(|quad| (extent(quad).1.x - 2.0).abs() < 1e-6),
        "merged across both cells"
    );
}

#[test]
fn stair_sub_quads_sample_the_texture_where_they_sit() {
    // A stair's quads must tile with the whole blocks around them, so the
    // texture coordinates of a sub-block quad are its position within the
    // cell — exactly as a slab's are.
    let q = mesh_single(TestBlock::Stair);
    let face = AlignedFace::PosZ;
    for quad in &q.faces[face.index()] {
        let positions = quad.positions(face, TestBlock::Stair.shape());
        let uvs = quad.texture_coordinates(face, TestBlock::Stair.shape(), Axis::X, true);
        for (p, uv) in positions.iter().zip(uvs) {
            // u follows x; v is flipped so it follows 1 - y.
            assert!((uv.x - p.x).abs() < 1e-6, "{p:?} {uv:?}");
            assert!((uv.y - (1.0 - p.y)).abs() < 1e-6, "{p:?} {uv:?}");
        }
    }
}

#[test]
fn a_stair_occludes_ambient_light_only_where_it_is_full() {
    // The AO pass asks a neighbor whether it fills the face it would
    // darken through. A stair fills its floor and its back and nothing
    // else — observed through a stone's top-face AO rather than a private
    // function.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum B {
        Air,
        Stone,
        Stair,
        Corbel,
    }
    impl Block for B {
        type TransparentGroup = ();
        type Light = u8;
        fn shape(&self) -> Shape {
            match self {
                B::Stair => TestBlock::Stair.shape(),
                B::Corbel => TestBlock::Corbel.shape(),
                _ => Shape::WholeBlock,
            }
        }
        fn cull_mode(&self) -> CullMode {
            match self {
                B::Air => CullMode::Empty,
                _ => CullMode::Opaque,
            }
        }
    }
    let stone_top_ao = |beside: B| {
        // The stone's top face samples the plane above it; a block
        // diagonally up from it darkens the two vertices on that side if
        // it fills its underside.
        let mut chunk = PaddedChunk16::<B>::new_filled(B::Air);
        chunk.set(glam::UVec3::new(0, 0, 0), B::Stone);
        chunk.set(glam::UVec3::new(1, 1, 0), beside);
        let q = mesh_chunk(&chunk, false);
        q.faces[AlignedFace::PosY.index()]
            .iter()
            .find(|quad| quad.voxel_position(AlignedFace::PosY) == glam::UVec3::ZERO)
            .expect("the stone's top")
            .ao
    };
    // A floor stair's underside is its full floor: it darkens.
    assert!(
        stone_top_ao(B::Stair).iter().any(|&ao| ao < 3),
        "a stair's floor casts AO"
    );
    // A corbel's underside is only its step: it does not.
    assert_eq!(
        stone_top_ao(B::Corbel),
        [3; 4],
        "a corbel that stops short of the plane casts no AO"
    );
    // And a full block darkens as it always did.
    assert!(stone_top_ao(B::Stone).iter().any(|&ao| ao < 3));
}

/// The AO of every quad on `face` of the stair at (2, 1, 2), keyed by the
/// quad's own extent so the two a stair shows can be told apart.
fn stair_face_ao(blocks: &[(u32, u32, u32, TestBlock)], face: AlignedFace) -> Vec<(Vec3, [u8; 4])> {
    let q = mesh_with(blocks);
    q.faces[face.index()]
        .iter()
        .filter(|quad| quad.voxel_position(face) == glam::UVec3::new(2, 1, 2))
        .map(|quad| {
            (
                extent(&quad.positions(face, TestBlock::Stair.shape())).0,
                quad.ao,
            )
        })
        .collect()
}

#[test]
fn a_treads_ao_is_read_where_the_tread_is() {
    // Stone against the stair's back and both its sides. The top of the
    // cell is dark along the back edge and lit along the front, and the
    // tread is only the front half of it: its inner edge stands halfway
    // across that gradient, not at the dark end of it.
    //
    // Handing the tread the cell's own corner values put the full
    // darkening of the stone behind the step onto the tread's inner edge,
    // a cell away from it — a dark band down the middle of every stair
    // with anything standing near it.
    let ao = stair_face_ao(
        &[
            (2, 1, 2, TestBlock::Stair),
            (2, 1, 1, TestBlock::Stone),
            (1, 1, 2, TestBlock::Stone),
            (3, 1, 2, TestBlock::Stone),
        ],
        AlignedFace::PosY,
    );
    // The tread is the half at half height; the step's top is at the top.
    let tread = ao
        .iter()
        .find(|(min, _)| (min.y - 1.5).abs() < 1e-6)
        .expect("the tread");
    let step = ao
        .iter()
        .find(|(min, _)| (min.y - 2.0).abs() < 1e-6)
        .expect("the step's top");
    // The cell's corners are [0, 2, 2, 0]; halfway along u they are 1.
    assert_eq!(tread.1, [1, 2, 2, 1], "the tread took the cell's corners");
    // The step's top looks at the plane above, which nothing reaches.
    assert_eq!(step.1, [3; 4], "the step's top darkened out of nowhere");
}

#[test]
fn a_steps_ao_is_read_where_the_step_is() {
    // The worst case, because it is offset both ways: the quarter a stair
    // shows on its side, which covers the far half of the upper half. A
    // stone under the neighboring cell darkens the bottom of that side —
    // the slab's strip, which touches it — and must not reach the step
    // standing a half cell above and a half cell back from it.
    let ao = stair_face_ao(
        &[(2, 1, 2, TestBlock::Stair), (1, 0, 2, TestBlock::Stone)],
        AlignedFace::NegX,
    );
    let slab = ao
        .iter()
        .find(|(min, _)| (min.y - 1.0).abs() < 1e-6)
        .expect("the slab's strip");
    let step = ao
        .iter()
        .find(|(min, _)| (min.y - 1.5).abs() < 1e-6)
        .expect("the step's quarter");
    // The strip stands on the darkened edge, so it keeps it.
    assert_eq!(slab.1, [2, 2, 3, 3], "the strip lost the stone below it");
    assert_eq!(step.1, [3; 4], "the step took the strip's darkening");
}
