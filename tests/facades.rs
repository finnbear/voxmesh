mod common;

use common::*;
use voxmesh::*;

#[test]
fn single_facade_produces_one_quad() {
    let q = mesh_single(TestBlock::Ladder);
    // Only the facade's own face (PosX) should have a quad.
    assert_eq!(face_count(&q, AlignedFace::PosX), 1);
    for face in [
        AlignedFace::NegX,
        AlignedFace::PosY,
        AlignedFace::NegY,
        AlignedFace::PosZ,
        AlignedFace::NegZ,
    ] {
        assert_eq!(face_count(&q, face), 0, "face {:?} should be empty", face);
    }
    // No diagonal quads.
    for diag in DiagonalFace::ALL {
        assert_eq!(q.diagonals[diag.index()].len(), 0);
    }
    assert_eq!(q.total(), 1);
}

#[test]
fn facade_negy_produces_one_quad() {
    let q = mesh_single(TestBlock::Rail);
    assert_eq!(face_count(&q, AlignedFace::NegY), 1);
    for face in [
        AlignedFace::PosX,
        AlignedFace::NegX,
        AlignedFace::PosY,
        AlignedFace::PosZ,
        AlignedFace::NegZ,
    ] {
        assert_eq!(face_count(&q, face), 0, "face {:?} should be empty", face);
    }
    assert_eq!(q.total(), 1);
}

#[test]
fn facade_mesh_block_matches_mesh_chunk() {
    let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
    chunk.set(glam::UVec3::ZERO, TestBlock::Ladder);
    let from_chunk = mesh_chunk(&chunk, true);
    let from_block = mesh_block(&TestBlock::Ladder, ());
    assert_eq!(from_chunk.total(), from_block.total());
    for face in AlignedFace::ALL {
        assert_eq!(
            from_chunk.faces[face.index()].len(),
            from_block.faces[face.index()].len(),
            "face {:?}",
            face
        );
    }
}

#[test]
fn facade_posx_is_offset_from_face() {
    let q = mesh_block(&TestBlock::Ladder, ());
    let quad = &q.faces[AlignedFace::PosX.index()][0];
    let verts = quad.positions(AlignedFace::PosX, TestBlock::Ladder.shape());

    // All vertices should have x = 15/16 (offset 1/16 inward from +X face).
    for (i, v) in verts.iter().enumerate() {
        assert!(
            (v.x - 15.0 / 16.0).abs() < 1e-6,
            "vertex {i} x should be 15/16, got {}",
            v.x
        );
    }
}

#[test]
fn facade_negy_is_offset_from_face() {
    let q = mesh_block(&TestBlock::Rail, ());
    let quad = &q.faces[AlignedFace::NegY.index()][0];
    let verts = quad.positions(AlignedFace::NegY, TestBlock::Rail.shape());

    // All vertices should have y = 1/16 (offset 1/16 inward from -Y face).
    for (i, v) in verts.iter().enumerate() {
        assert!(
            (v.y - 1.0 / 16.0).abs() < 1e-6,
            "vertex {i} y should be 1/16, got {}",
            v.y
        );
    }
}

#[test]
fn facade_does_not_cull_neighbor() {
    // A stone block adjacent to a facade should still have all 6 faces.
    let q = mesh_with(&[(0, 0, 0, TestBlock::Ladder), (1, 0, 0, TestBlock::Stone)]);
    // Stone's NegX face (toward the facade) should still be present.
    let stone_neg_x = q.faces[AlignedFace::NegX.index()].iter().any(|quad| {
        let verts = quad.positions(AlignedFace::NegX, TestBlock::Ladder.shape());
        (verts[0].x - 1.0).abs() < 1e-6
    });
    assert!(stone_neg_x, "stone NegX face should be present");
}

#[test]
fn opaque_neighbor_does_not_cull_facade() {
    // An opaque block next to a facade should not cull it (facade is offset inward).
    let q = mesh_with(&[(0, 0, 0, TestBlock::Ladder), (1, 0, 0, TestBlock::Stone)]);
    // The ladder's PosX face should still be present.
    let has_ladder = q.faces[AlignedFace::PosX.index()].iter().any(|quad| {
        let verts = quad.positions(AlignedFace::PosX, TestBlock::Ladder.shape());
        (verts[0].x - 15.0 / 16.0).abs() < 1e-6
    });
    assert!(has_ladder, "ladder PosX facade should be present");
}

#[test]
fn identical_facades_merge_along_v() {
    // Three ladders stacked vertically (same face PosX) should merge into one quad.
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::Ladder),
        (0, 1, 0, TestBlock::Ladder),
        (0, 2, 0, TestBlock::Ladder),
    ]);
    assert_eq!(
        face_count(&q, AlignedFace::PosX),
        1,
        "stacked facades should merge into 1 quad"
    );
}

#[test]
fn identical_facades_merge_along_u() {
    // Three ladders in a row along Z (same face PosX) should merge into one quad.
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::Ladder),
        (0, 0, 1, TestBlock::Ladder),
        (0, 0, 2, TestBlock::Ladder),
    ]);
    assert_eq!(
        face_count(&q, AlignedFace::PosX),
        1,
        "row of facades should merge into 1 quad"
    );
}

#[test]
fn different_face_facades_do_not_merge() {
    // A PosX facade next to a NegY facade should not merge (different face lists).
    let q = mesh_with(&[(0, 0, 0, TestBlock::Ladder), (0, 0, 1, TestBlock::Rail)]);
    assert_eq!(face_count(&q, AlignedFace::PosX), 1);
    assert_eq!(face_count(&q, AlignedFace::NegY), 1);
    assert_eq!(q.total(), 2);
}

#[test]
fn facade_voxel_position_is_correct() {
    let q = mesh_single(TestBlock::Ladder);
    let quad = &q.faces[AlignedFace::PosX.index()][0];
    let vp = quad.voxel_position(AlignedFace::PosX);
    assert_eq!(vp, glam::UVec3::ZERO);

    let q = mesh_single(TestBlock::Rail);
    let quad = &q.faces[AlignedFace::NegY.index()][0];
    let vp = quad.voxel_position(AlignedFace::NegY);
    assert_eq!(vp, glam::UVec3::ZERO);
}

#[test]
fn facade_texture_coordinates_span_one() {
    let q = mesh_block(&TestBlock::Ladder, ());
    let quad = &q.faces[AlignedFace::PosX.index()][0];
    let uvs = quad.texture_coordinates(AlignedFace::PosX, Axis::X, false);

    let u_min = uvs.iter().map(|uv| uv.x).fold(f32::INFINITY, f32::min);
    let u_max = uvs.iter().map(|uv| uv.x).fold(f32::NEG_INFINITY, f32::max);
    let v_min = uvs.iter().map(|uv| uv.y).fold(f32::INFINITY, f32::min);
    let v_max = uvs.iter().map(|uv| uv.y).fold(f32::NEG_INFINITY, f32::max);

    assert!((u_min).abs() < 1e-6, "u_min should be 0");
    assert!((u_max - 1.0).abs() < 1e-6, "u_max should be 1");
    assert!((v_min).abs() < 1e-6, "v_min should be 0");
    assert!((v_max - 1.0).abs() < 1e-6, "v_max should be 1");
}
