mod common;

use common::*;
use voxmesh::*;

#[test]
fn single_inset_produces_six_quads() {
    let q = mesh_single(TestBlock::Cactus);
    for face in Face::ALL {
        assert_eq!(
            face_count(&q, face),
            1,
            "face {:?} should have 1 quad",
            face
        );
    }
    // No diagonal quads.
    for diag in DiagonalFace::ALL {
        assert_eq!(q.diagonals[diag.index()].len(), 0);
    }
    assert_eq!(q.total(), 6);
}

#[test]
fn inset_block_faces_matches_greedy_mesh() {
    let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
    chunk.set(glam::UVec3::ZERO, TestBlock::Cactus);
    let from_chunk = greedy_mesh(&chunk);
    let from_block = block_faces(&TestBlock::Cactus);
    assert_eq!(from_chunk.total(), from_block.total());
    for face in Face::ALL {
        assert_eq!(
            from_chunk.faces[face.index()].len(),
            from_block.faces[face.index()].len(),
            "face {:?}",
            face
        );
    }
}

#[test]
fn inset_side_faces_are_offset() {
    let q = block_faces(&TestBlock::Cactus);

    // PosX: all vertices should have x = 15/16.
    assert_face_on_plane(&q, Face::PosX, 0, 15.0 / 16.0);
    // NegX: all vertices should have x = 1/16.
    assert_face_on_plane(&q, Face::NegX, 0, 1.0 / 16.0);
    // PosZ: all vertices should have z = 15/16.
    assert_face_on_plane(&q, Face::PosZ, 2, 15.0 / 16.0);
    // NegZ: all vertices should have z = 1/16.
    assert_face_on_plane(&q, Face::NegZ, 2, 1.0 / 16.0);
}

#[test]
fn inset_top_bottom_at_boundary() {
    let q = block_faces(&TestBlock::Cactus);

    // PosY: all vertices should have y = 1.0.
    assert_face_on_plane(&q, Face::PosY, 1, 1.0);
    // NegY: all vertices should have y = 0.0.
    assert_face_on_plane(&q, Face::NegY, 1, 0.0);
}

#[test]
fn opaque_neighbor_culls_inset_top_bottom() {
    // Stone above cactus should cull the cactus PosY face.
    let q = mesh_with(&[(0, 0, 0, TestBlock::Cactus), (0, 1, 0, TestBlock::Stone)]);

    // Cactus PosY should be culled (stone covers it).
    let cactus_pos_y = q.faces[Face::PosY.index()].iter().any(|quad| {
        let verts = quad.positions(Face::PosY, TestBlock::Cactus.shape());
        (verts[0].y - 1.0).abs() < 1e-6
    });
    assert!(!cactus_pos_y, "cactus PosY should be culled by stone above");
}

#[test]
fn opaque_neighbor_does_not_cull_inset_side() {
    // Stone adjacent to cactus on +X side should NOT cull the cactus PosX face.
    let q = mesh_with(&[(0, 0, 0, TestBlock::Cactus), (1, 0, 0, TestBlock::Stone)]);

    let has_inset_pos_x = q.faces[Face::PosX.index()].iter().any(|quad| {
        let verts = quad.positions(Face::PosX, TestBlock::Cactus.shape());
        (verts[0].x - 15.0 / 16.0).abs() < 1e-6
    });
    assert!(
        has_inset_pos_x,
        "cactus PosX should NOT be culled by adjacent stone"
    );
}

#[test]
fn inset_does_not_cull_neighbor_side() {
    // Stone's NegX face (toward the cactus) should still be visible.
    let q = mesh_with(&[(0, 0, 0, TestBlock::Cactus), (1, 0, 0, TestBlock::Stone)]);

    let stone_neg_x = q.faces[Face::NegX.index()].iter().any(|quad| {
        let verts = quad.positions(Face::NegX, TestBlock::Cactus.shape());
        (verts[0].x - 1.0).abs() < 1e-6
    });
    assert!(
        stone_neg_x,
        "stone NegX face should be visible (cactus side is recessed)"
    );
}

#[test]
fn inset_culls_neighbor_top_bottom() {
    // Cactus above stone should cull stone's PosY face (inset top/bottom are flush).
    let q = mesh_with(&[(0, 1, 0, TestBlock::Cactus), (0, 0, 0, TestBlock::Stone)]);

    // Stone's PosY (at y=1) should be culled by the cactus NegY above it.
    let stone_pos_y = q.faces[Face::PosY.index()].iter().any(|quad| {
        let verts = quad.positions(Face::PosY, TestBlock::Cactus.shape());
        (verts[0].y - 1.0).abs() < 1e-6
    });
    assert!(!stone_pos_y, "stone PosY should be culled by cactus above");
}

#[test]
fn identical_insets_merge_horizontally() {
    // Three cactus blocks in a row along Z should merge side faces.
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::Cactus),
        (0, 0, 1, TestBlock::Cactus),
        (0, 0, 2, TestBlock::Cactus),
    ]);
    // PosX face: u=Z, so 3 blocks along Z should merge into 1 quad.
    assert_eq!(
        face_count(&q, Face::PosX),
        1,
        "row of inset blocks should merge PosX face"
    );
}

#[test]
fn identical_insets_merge_vertically() {
    // Three cactus blocks stacked vertically should merge side faces.
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::Cactus),
        (0, 1, 0, TestBlock::Cactus),
        (0, 2, 0, TestBlock::Cactus),
    ]);
    // PosX face: v=Y, so 3 blocks along Y should merge into 1 quad.
    assert_eq!(
        face_count(&q, Face::PosX),
        1,
        "stacked inset blocks should merge PosX face"
    );
}

#[test]
fn inset_voxel_position_is_correct() {
    let q = mesh_single(TestBlock::Cactus);
    for face in Face::ALL {
        let quad = &q.faces[face.index()][0];
        let vp = quad.voxel_position(face);
        assert_eq!(vp, glam::UVec3::ZERO, "face {:?}", face);
    }
}

#[test]
fn inset_texture_coordinates_span_one() {
    let q = block_faces(&TestBlock::Cactus);
    for face in Face::ALL {
        let quad = &q.faces[face.index()][0];
        let uvs = quad.texture_coordinates(face, Axis::X, false);

        let u_min = uvs.iter().map(|uv| uv.x).fold(f32::INFINITY, f32::min);
        let u_max = uvs.iter().map(|uv| uv.x).fold(f32::NEG_INFINITY, f32::max);
        let v_min = uvs.iter().map(|uv| uv.y).fold(f32::INFINITY, f32::min);
        let v_max = uvs.iter().map(|uv| uv.y).fold(f32::NEG_INFINITY, f32::max);

        assert!((u_min).abs() < 1e-6, "face {:?}: u_min should be 0", face);
        assert!(
            (u_max - 1.0).abs() < 1e-6,
            "face {:?}: u_max should be 1",
            face
        );
        assert!((v_min).abs() < 1e-6, "face {:?}: v_min should be 0", face);
        assert!(
            (v_max - 1.0).abs() < 1e-6,
            "face {:?}: v_max should be 1",
            face
        );
    }
}
