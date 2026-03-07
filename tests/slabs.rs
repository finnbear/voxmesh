mod common;

use common::*;
use voxmesh::*;

#[test]
fn upper_slab_produces_six_faces() {
    let q = mesh_single(TestBlock::UpperSlab);
    assert_eq!(q.total(), 6);
}

#[test]
fn upper_slab_posy_is_at_block_top() {
    let q = mesh_single(TestBlock::UpperSlab);
    assert_face_on_plane(&q, AlignedFace::PosY, 1, 1.0);
}

#[test]
fn upper_slab_negy_is_inset() {
    let q = mesh_single(TestBlock::UpperSlab);
    assert_face_on_plane(&q, AlignedFace::NegY, 1, 0.5);
}

#[test]
fn upper_slab_side_face_is_half_height() {
    let q = mesh_single(TestBlock::UpperSlab);
    let (y_min, y_max) = face_vertex_range(&q, AlignedFace::PosX, 1);
    assert!(
        (y_min - 0.5).abs() < 1e-6,
        "side face y_min should be 0.5, got {y_min}"
    );
    assert!(
        (y_max - 1.0).abs() < 1e-6,
        "side face y_max should be 1.0, got {y_max}"
    );
}

#[test]
fn lower_slab_negy_is_flush_at_bottom() {
    let q = mesh_single(TestBlock::LowerSlab);
    assert_face_on_plane(&q, AlignedFace::NegY, 1, 0.0);
}

#[test]
fn lower_slab_posy_is_inset_at_half() {
    let q = mesh_single(TestBlock::LowerSlab);
    assert_face_on_plane(&q, AlignedFace::PosY, 1, 0.5);
}

#[test]
fn opaque_block_above_upper_slab_culls_flush_face() {
    let q = mesh_with(&[(0, 0, 0, TestBlock::UpperSlab), (0, 1, 0, TestBlock::Stone)]);
    // Stone culls slab's PosY (flush), slab culls stone's NegY.
    // Only stone's PosY remains on that face direction.
    assert_eq!(face_count(&q, AlignedFace::PosY), 1);
}

#[test]
fn slab_inset_face_never_culled() {
    let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
    chunk.set(glam::UVec3::ZERO, TestBlock::UpperSlab);
    // Place stone in padding below slab (padded y=0).
    chunk.set_padded(glam::UVec3::new(1, 0, 1), TestBlock::Stone);
    let q = mesh_chunk(&chunk, true);

    // The slab's NegY (inset) face at y=0.5 must still be present.
    let has_inset = q.faces[AlignedFace::NegY.index()].iter().any(|quad| {
        let verts = quad.positions(AlignedFace::NegY, TestBlock::UpperSlab.shape());
        (verts[0].y - 0.5).abs() < 1e-6
    });
    assert!(has_inset, "slab inset NegY face should be present at y=0.5");
}

/// Slab side-face UVs must match the corresponding portion of a whole
/// block's side-face UVs (the texture should not shift between a block
/// and an adjacent slab).
#[test]
fn slab_side_face_uvs_match_whole_block() {
    let stone_q = mesh_single(TestBlock::Stone);
    let upper_q = mesh_single(TestBlock::UpperSlab);
    let lower_q = mesh_single(TestBlock::LowerSlab);

    // Side faces to check: PosX, NegX, PosZ, NegZ.
    // For PosY slabs the slab axis maps to the V tangent on these faces.
    let side_faces = [
        AlignedFace::PosX,
        AlignedFace::NegX,
        AlignedFace::PosZ,
        AlignedFace::NegZ,
    ];

    for &face in &side_faces {
        let stone_uvs = stone_q.faces[face.index()][0].texture_coordinates(face, Axis::X, false);
        let upper_uvs = upper_q.faces[face.index()][0].texture_coordinates(face, Axis::X, false);
        let lower_uvs = lower_q.faces[face.index()][0].texture_coordinates(face, Axis::X, false);

        // Stone V range is [0, 1]. Upper slab should be [0.5, 1.0],
        // lower slab should be [0.0, 0.5].
        let stone_v_min = stone_uvs.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let stone_v_max = stone_uvs
            .iter()
            .map(|v| v.y)
            .fold(f32::NEG_INFINITY, f32::max);

        let upper_v_min = upper_uvs.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let upper_v_max = upper_uvs
            .iter()
            .map(|v| v.y)
            .fold(f32::NEG_INFINITY, f32::max);

        let lower_v_min = lower_uvs.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let lower_v_max = lower_uvs
            .iter()
            .map(|v| v.y)
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            (stone_v_min).abs() < 1e-6,
            "face {face:?}: stone v_min should be 0.0, got {stone_v_min}"
        );
        assert!(
            (stone_v_max - 1.0).abs() < 1e-6,
            "face {face:?}: stone v_max should be 1.0, got {stone_v_max}"
        );

        assert!(
            (upper_v_min - 0.5).abs() < 1e-6,
            "face {face:?}: upper slab v_min should be 0.5, got {upper_v_min}"
        );
        assert!(
            (upper_v_max - 1.0).abs() < 1e-6,
            "face {face:?}: upper slab v_max should be 1.0, got {upper_v_max}"
        );

        assert!(
            (lower_v_min).abs() < 1e-6,
            "face {face:?}: lower slab v_min should be 0.0, got {lower_v_min}"
        );
        assert!(
            (lower_v_max - 0.5).abs() < 1e-6,
            "face {face:?}: lower slab v_max should be 0.5, got {lower_v_max}"
        );

        // U range should be identical to the whole block (slab doesn't
        // affect the non-slab tangent axis).
        let stone_u_min = stone_uvs.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
        let stone_u_max = stone_uvs
            .iter()
            .map(|v| v.x)
            .fold(f32::NEG_INFINITY, f32::max);
        let upper_u_min = upper_uvs.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
        let upper_u_max = upper_uvs
            .iter()
            .map(|v| v.x)
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            (upper_u_min - stone_u_min).abs() < 1e-6,
            "face {face:?}: upper slab u_min should match stone, got {upper_u_min} vs {stone_u_min}"
        );
        assert!(
            (upper_u_max - stone_u_max).abs() < 1e-6,
            "face {face:?}: upper slab u_max should match stone, got {upper_u_max} vs {stone_u_max}"
        );
    }
}

/// Per-vertex UVs of a slab side face must equal the corresponding
/// vertices of a whole block when placed at the same position, for all
/// flip combinations.
#[test]
fn slab_side_face_per_vertex_uvs_match_whole_block() {
    let stone_q = mesh_single(TestBlock::Stone);
    let upper_q = mesh_single(TestBlock::UpperSlab);

    let side_faces = [
        AlignedFace::PosX,
        AlignedFace::NegX,
        AlignedFace::PosZ,
        AlignedFace::NegZ,
    ];

    for &face in &side_faces {
        for &u_flip in &[Axis::X, Axis::Y, Axis::Z] {
            for &flip_v in &[false, true] {
                let stone_uvs =
                    stone_q.faces[face.index()][0].texture_coordinates(face, u_flip, flip_v);
                let upper_uvs =
                    upper_q.faces[face.index()][0].texture_coordinates(face, u_flip, flip_v);

                // Find the stone vertices that fall within the upper
                // slab's Y range [0.5, 1.0] and verify the UVs match.
                let stone_pos = stone_q.faces[face.index()][0].positions(face, Shape::WholeBlock);
                let upper_pos =
                    upper_q.faces[face.index()][0].positions(face, TestBlock::UpperSlab.shape());

                // For each slab vertex, find the stone vertex at the
                // same position and compare UVs.
                for (si, slab_p) in upper_pos.iter().enumerate() {
                    for (bi, block_p) in stone_pos.iter().enumerate() {
                        if (*slab_p - *block_p).length() < 1e-6 {
                            let diff = (upper_uvs[si] - stone_uvs[bi]).length();
                            assert!(
                                diff < 1e-6,
                                "face {face:?} u_flip={u_flip:?} flip_v={flip_v}: \
                                 slab vertex {si} UV {:?} != stone vertex {bi} UV {:?}",
                                upper_uvs[si],
                                stone_uvs[bi],
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Stacked lower-slabs must not merge their side faces along the slab axis.
/// Each slab only occupies the bottom half of its cell, so merging would
/// produce a quad that incorrectly spans the empty upper halves.
#[test]
fn stacked_lower_slabs_side_faces_not_merged_vertically() {
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::LowerSlab),
        (0, 1, 0, TestBlock::LowerSlab),
    ]);

    // Each slab should produce its own PosX side face (2 quads, not 1).
    let side_quads = &q.faces[AlignedFace::PosX.index()];
    assert_eq!(
        side_quads.len(),
        2,
        "two stacked lower-slabs should have 2 separate PosX side quads, got {}",
        side_quads.len(),
    );

    // Each side quad should span only half a block in Y.
    for quad in side_quads {
        let verts = quad.positions(AlignedFace::PosX, TestBlock::UpperSlab.shape());
        let y_min = verts.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let y_max = verts.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max);
        let height = y_max - y_min;
        assert!(
            (height - 0.5).abs() < 1e-6,
            "each slab side quad should be 0.5 tall, got {height}",
        );
    }
}
