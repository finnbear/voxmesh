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
    assert_face_on_plane(&q, Face::PosY, 1, 1.0);
}

#[test]
fn upper_slab_negy_is_inset() {
    let q = mesh_single(TestBlock::UpperSlab);
    assert_face_on_plane(&q, Face::NegY, 1, 0.5);
}

#[test]
fn upper_slab_side_face_is_half_height() {
    let q = mesh_single(TestBlock::UpperSlab);
    let (y_min, y_max) = face_vertex_range(&q, Face::PosX, 1);
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
    assert_face_on_plane(&q, Face::NegY, 1, 0.0);
}

#[test]
fn lower_slab_posy_is_inset_at_half() {
    let q = mesh_single(TestBlock::LowerSlab);
    assert_face_on_plane(&q, Face::PosY, 1, 0.5);
}

#[test]
fn opaque_block_above_upper_slab_culls_flush_face() {
    let q = mesh_with(&[(0, 0, 0, TestBlock::UpperSlab), (0, 1, 0, TestBlock::Stone)]);
    // Stone culls slab's PosY (flush), slab culls stone's NegY.
    // Only stone's PosY remains on that face direction.
    assert_eq!(face_count(&q, Face::PosY), 1);
}

#[test]
fn slab_inset_face_never_culled() {
    let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
    chunk.set(0, 0, 0, TestBlock::UpperSlab);
    // Place stone in padding below slab (padded y=0).
    chunk.set_padded(1, 0, 1, TestBlock::Stone);
    let q = greedy_mesh(&chunk);

    // The slab's NegY (inset) face at y=0.5 must still be present.
    let has_inset = q.faces[Face::NegY.index()].iter().any(|quad| {
        let verts = quad.positions(Face::NegY);
        (verts[0].y - 0.5).abs() < 1e-6
    });
    assert!(has_inset, "slab inset NegY face should be present at y=0.5");
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
    let side_quads = &q.faces[Face::PosX.index()];
    assert_eq!(
        side_quads.len(),
        2,
        "two stacked lower-slabs should have 2 separate PosX side quads, got {}",
        side_quads.len(),
    );

    // Each side quad should span only half a block in Y.
    for quad in side_quads {
        let verts = quad.positions(Face::PosX);
        let y_min = verts.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let y_max = verts.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max);
        let height = y_max - y_min;
        assert!(
            (height - 0.5).abs() < 1e-6,
            "each slab side quad should be 0.5 tall, got {height}",
        );
    }
}
