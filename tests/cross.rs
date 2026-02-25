mod common;

use common::*;
use voxmesh::*;

#[test]
fn single_cross_produces_two_diagonal_quads() {
    let q = mesh_single(TestBlock::SugarCane);
    // No axis-aligned faces.
    for face in Face::ALL {
        assert_eq!(face_count(&q, face), 0, "face {:?} should be empty", face);
    }
    // Two diagonal quads (one per DiagonalFace).
    for diag in DiagonalFace::ALL {
        assert_eq!(
            q.diagonals[diag.index()].len(),
            1,
            "diagonal {:?} should have 1 quad",
            diag
        );
    }
    assert_eq!(q.total(), 2);
}

#[test]
fn cross_block_faces_matches_greedy_mesh() {
    let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
    chunk.set(0, 0, 0, TestBlock::SugarCane);
    let from_chunk = greedy_mesh(&chunk);
    let from_block = block_faces(&TestBlock::SugarCane);
    assert_eq!(from_chunk.total(), from_block.total());
    for diag in DiagonalFace::ALL {
        assert_eq!(
            from_chunk.diagonals[diag.index()].len(),
            from_block.diagonals[diag.index()].len(),
            "diagonal {:?}",
            diag
        );
    }
}

#[test]
fn cross_does_not_produce_axis_aligned_faces() {
    let q = block_faces(&TestBlock::Cobweb);
    for face in Face::ALL {
        assert_eq!(face_count(&q, face), 0);
    }
    assert_eq!(q.total(), 2);
}

#[test]
fn stacked_cross_blocks_merge_vertically() {
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::SugarCane),
        (0, 1, 0, TestBlock::SugarCane),
        (0, 2, 0, TestBlock::SugarCane),
    ]);
    // Three stacked identical cross blocks should merge into one quad per diagonal.
    for diag in DiagonalFace::ALL {
        assert_eq!(
            q.diagonals[diag.index()].len(),
            1,
            "diagonal {:?} should merge into 1 quad",
            diag
        );
    }
    assert_eq!(q.total(), 2);
}

#[test]
fn different_cross_blocks_do_not_merge() {
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::SugarCane),
        (0, 1, 0, TestBlock::Cobweb),
    ]);
    // Different blocks should not merge.
    for diag in DiagonalFace::ALL {
        assert_eq!(
            q.diagonals[diag.index()].len(),
            2,
            "diagonal {:?} should have 2 quads",
            diag
        );
    }
    assert_eq!(q.total(), 4);
}

#[test]
fn cross_does_not_cull_adjacent_opaque_block() {
    let q = mesh_with(&[(0, 0, 0, TestBlock::SugarCane), (1, 0, 0, TestBlock::Stone)]);
    // Stone should still have all 6 faces (cross doesn't cull it).
    // The face toward the cross block (NegX) should be present.
    let stone_neg_x = q.faces[Face::NegX.index()].iter().any(|quad| {
        let verts = quad.positions(Face::NegX, 0, Face::NegY);
        (verts[0].x - 1.0).abs() < 1e-6
    });
    assert!(stone_neg_x, "stone NegX face should be present");
}

#[test]
fn cross_diagonal_positions_span_unit_block() {
    let q = block_faces(&TestBlock::SugarCane);

    for diag in DiagonalFace::ALL {
        let quad = &q.diagonals[diag.index()][0];
        let verts = quad.positions(diag, 0, Face::NegY);

        // Y should span [0, 1].
        let y_min = verts.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let y_max = verts.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (y_min).abs() < 1e-6,
            "diagonal {:?} y_min should be 0, got {y_min}",
            diag
        );
        assert!(
            (y_max - 1.0).abs() < 1e-6,
            "diagonal {:?} y_max should be 1, got {y_max}",
            diag
        );

        // All vertices should be within the unit cube [0,1]^3.
        for (i, v) in verts.iter().enumerate() {
            assert!(
                v.x >= -1e-6 && v.x <= 1.0 + 1e-6,
                "diagonal {:?} vertex {i} x={} out of [0,1]",
                diag,
                v.x
            );
            assert!(
                v.y >= -1e-6 && v.y <= 1.0 + 1e-6,
                "diagonal {:?} vertex {i} y={} out of [0,1]",
                diag,
                v.y
            );
            assert!(
                v.z >= -1e-6 && v.z <= 1.0 + 1e-6,
                "diagonal {:?} vertex {i} z={} out of [0,1]",
                diag,
                v.z
            );
        }
    }
}

#[test]
fn merged_cross_diagonal_height_spans_multiple_blocks() {
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::SugarCane),
        (0, 1, 0, TestBlock::SugarCane),
    ]);

    for diag in DiagonalFace::ALL {
        let quad = &q.diagonals[diag.index()][0];
        let verts = quad.positions(diag, 0, Face::NegY);

        let y_min = verts.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let y_max = verts.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (y_min).abs() < 1e-6,
            "diagonal {:?} y_min should be 0, got {y_min}",
            diag
        );
        assert!(
            (y_max - 2.0).abs() < 1e-6,
            "diagonal {:?} y_max should be 2.0, got {y_max}",
            diag
        );
    }
}

#[test]
fn stretched_cross_extends_beyond_block_boundary() {
    let q = block_faces(&TestBlock::Cobweb);

    // Cobweb has stretch=4 (4/16 = 0.25 extra on each side).
    for diag in DiagonalFace::ALL {
        let quad = &q.diagonals[diag.index()][0];
        let verts = quad.positions(diag, 4, Face::NegY);

        // The diagonal extent should be wider than 1 block.
        let x_min = verts.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
        let x_max = verts.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max);
        let x_span = x_max - x_min;
        // With stretch=0, diagonal A spans x=[0,1] (span=1.0).
        // With stretch=4, it should be wider: span = 1.0 + 2*0.25 = 1.5.
        assert!(
            x_span > 1.0 + 1e-6,
            "diagonal {:?} x span should be > 1.0 with stretch, got {x_span}",
            diag
        );
    }
}

#[test]
fn cross_horizontal_neighbors_do_not_merge() {
    // Two cross blocks side by side should not merge (only vertical merging).
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::SugarCane),
        (1, 0, 0, TestBlock::SugarCane),
    ]);
    for diag in DiagonalFace::ALL {
        assert_eq!(
            q.diagonals[diag.index()].len(),
            2,
            "diagonal {:?} should have 2 separate quads for horizontally adjacent cross blocks",
            diag
        );
    }
    assert_eq!(q.total(), 4);
}

#[test]
fn cross_texture_coordinates_span_one_by_one() {
    let q = block_faces(&TestBlock::SugarCane);
    for diag in DiagonalFace::ALL {
        let quad = &q.diagonals[diag.index()][0];
        let uvs = quad.texture_coordinates(diag, Axis::X, false);
        assert_eq!(uvs[0], glam::Vec2::new(0.0, 0.0));
        assert_eq!(uvs[1], glam::Vec2::new(1.0, 0.0));
        assert_eq!(uvs[2], glam::Vec2::new(1.0, 1.0));
        assert_eq!(uvs[3], glam::Vec2::new(0.0, 1.0));
    }
}

#[test]
fn merged_cross_texture_coordinates_scale_vertically() {
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::SugarCane),
        (0, 1, 0, TestBlock::SugarCane),
        (0, 2, 0, TestBlock::SugarCane),
    ]);
    for diag in DiagonalFace::ALL {
        let quad = &q.diagonals[diag.index()][0];
        let uvs = quad.texture_coordinates(diag, Axis::X, false);
        // v_size should be 3.0 (3 blocks tall).
        assert_eq!(uvs[2], glam::Vec2::new(1.0, 3.0));
        assert_eq!(uvs[3], glam::Vec2::new(0.0, 3.0));
    }
}

// ---- Root face tests ----

#[test]
fn cross_posy_root_crosses_in_xz_plane() {
    // PosY root: same crossing plane as NegY (XZ), merge along Y.
    let q = block_faces(&TestBlock::Chain);
    assert_eq!(q.total(), 2);

    for diag in DiagonalFace::ALL {
        let quad = &q.diagonals[diag.index()][0];
        let verts = quad.positions(diag, 0, Face::PosY);

        // Y (merge axis) should span [0, 1].
        let y_min = verts.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let y_max = verts.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max);
        assert!((y_min).abs() < 1e-6, "y_min should be 0, got {y_min}");
        assert!((y_max - 1.0).abs() < 1e-6, "y_max should be 1, got {y_max}");

        // All vertices within the unit cube.
        for v in &verts {
            assert!(v.x >= -1e-6 && v.x <= 1.0 + 1e-6);
            assert!(v.z >= -1e-6 && v.z <= 1.0 + 1e-6);
        }
    }
}

#[test]
fn cross_posx_root_crosses_in_yz_plane() {
    // PosX root: crossing in YZ, merge along X.
    let q = block_faces(&TestBlock::Vine);
    assert_eq!(q.total(), 2);

    for diag in DiagonalFace::ALL {
        let quad = &q.diagonals[diag.index()][0];
        let verts = quad.positions(diag, 0, Face::PosX);

        // X (merge axis) should span [0, 1].
        let x_min = verts.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
        let x_max = verts.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max);
        assert!((x_min).abs() < 1e-6, "x_min should be 0, got {x_min}");
        assert!((x_max - 1.0).abs() < 1e-6, "x_max should be 1, got {x_max}");

        // Crossing axes (Y, Z) should be centered around 0.5.
        for v in &verts {
            assert!(v.y >= -1e-6 && v.y <= 1.0 + 1e-6);
            assert!(v.z >= -1e-6 && v.z <= 1.0 + 1e-6);
        }
    }
}

#[test]
fn cross_posx_root_merges_along_x() {
    // Three vine blocks in a row along X should merge.
    let q = mesh_with(&[
        (0, 0, 0, TestBlock::Vine),
        (1, 0, 0, TestBlock::Vine),
        (2, 0, 0, TestBlock::Vine),
    ]);
    for diag in DiagonalFace::ALL {
        assert_eq!(
            q.diagonals[diag.index()].len(),
            1,
            "diagonal {:?} should merge into 1 quad",
            diag
        );
    }
    assert_eq!(q.total(), 2);

    // Merged quad should span 3 blocks along X.
    let quad = &q.diagonals[DiagonalFace::A.index()][0];
    let verts = quad.positions(DiagonalFace::A, 0, Face::PosX);
    let x_min = verts.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
    let x_max = verts.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max);
    assert!((x_min).abs() < 1e-6, "x_min should be 0, got {x_min}");
    assert!(
        (x_max - 3.0).abs() < 1e-6,
        "x_max should be 3.0, got {x_max}"
    );
}

#[test]
fn cross_posx_root_does_not_merge_along_y() {
    // Vine blocks stacked along Y should NOT merge (Y is a crossing axis, not merge axis).
    let q = mesh_with(&[(0, 0, 0, TestBlock::Vine), (0, 1, 0, TestBlock::Vine)]);
    for diag in DiagonalFace::ALL {
        assert_eq!(
            q.diagonals[diag.index()].len(),
            2,
            "diagonal {:?} should have 2 quads (no merge along non-merge axis)",
            diag
        );
    }
    assert_eq!(q.total(), 4);
}

#[test]
fn different_root_face_crosses_do_not_merge() {
    // SugarCane (NegY root) and Chain (PosY root) stacked should not merge
    // (they are different blocks).
    let q = mesh_with(&[(0, 0, 0, TestBlock::SugarCane), (0, 1, 0, TestBlock::Chain)]);
    for diag in DiagonalFace::ALL {
        assert_eq!(
            q.diagonals[diag.index()].len(),
            2,
            "diagonal {:?} should have 2 quads",
            diag
        );
    }
    assert_eq!(q.total(), 4);
}
