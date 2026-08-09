#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

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
    let mut chunk = PaddedChunk16::new_filled(TestBlock::Air);
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

/// Every vertex of every face of every slab orientation must have UVs
/// that match the UV a whole block would produce at the same spatial
/// position, for all flip combinations.
#[test]
fn slab_side_face_uvs_match_whole_block() {
    let stone_q = mesh_single(TestBlock::Stone);

    let slabs: &[(TestBlock, &str)] = &[
        (TestBlock::UpperSlab, "UpperSlab(PosY)"),
        (TestBlock::LowerSlab, "LowerSlab(NegY)"),
        (TestBlock::PosXSlab, "PosXSlab"),
        (TestBlock::NegXSlab, "NegXSlab"),
        (TestBlock::PosZSlab, "PosZSlab"),
        (TestBlock::NegZSlab, "NegZSlab"),
    ];

    for &(slab_block, slab_name) in slabs {
        let slab_q = mesh_single(slab_block);

        for &face in &AlignedFace::ALL {
            let slab_quads = &slab_q.faces[face.index()];
            if slab_quads.is_empty() {
                continue;
            }

            for &u_flip in &[Axis::X, Axis::Y, Axis::Z] {
                for &flip_v in &[false, true] {
                    let stone_uvs = stone_q.faces[face.index()][0].texture_coordinates(
                        face,
                        Shape::WholeBlock,
                        u_flip,
                        flip_v,
                    );
                    let stone_pos =
                        stone_q.faces[face.index()][0].positions(face, Shape::WholeBlock);

                    let slab_uvs =
                        slab_quads[0].texture_coordinates(face, Shape::WholeBlock, u_flip, flip_v);
                    let slab_pos = slab_quads[0].positions(face, slab_block.shape());

                    for (vi, (slab_p, slab_uv)) in slab_pos.iter().zip(slab_uvs.iter()).enumerate()
                    {
                        let expected = bilinear_uv(slab_p, &stone_pos, &stone_uvs);
                        let diff = (*slab_uv - expected).length();
                        assert!(
                            diff < 1e-5,
                            "{slab_name} face {face:?} u_flip={u_flip:?} \
                             flip_v={flip_v}: vertex {vi} at {slab_p:?} \
                             UV {slab_uv:?} != expected {expected:?}",
                        );
                    }
                }
            }
        }
    }
}

/// Bilinear interpolation of UV from an axis-aligned quad.
fn bilinear_uv(p: &glam::Vec3, positions: &[glam::Vec3; 4], uvs: &[glam::Vec2; 4]) -> glam::Vec2 {
    let mut min_p = positions[0];
    let mut max_p = positions[0];
    for pos in &positions[1..] {
        min_p = min_p.min(*pos);
        max_p = max_p.max(*pos);
    }
    let range = max_p - min_p;
    let t = (*p - min_p) / range;

    let mut result = glam::Vec2::ZERO;
    for (i, pos) in positions.iter().enumerate() {
        let n = (*pos - min_p) / range;
        let mut weight = 1.0f32;
        for axis in 0..3 {
            if range[axis] > 1e-6 {
                if (n[axis] - 1.0).abs() < 1e-6 {
                    weight *= t[axis];
                } else {
                    weight *= 1.0 - t[axis];
                }
            }
        }
        result += weight * uvs[i];
    }
    result
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
