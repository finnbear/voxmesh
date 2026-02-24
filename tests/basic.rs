mod common;

use common::*;
use voxmesh::*;

#[test]
fn empty_chunk_produces_no_quads() {
    let chunk = PaddedChunk::new_filled(TestBlock::Air);
    let q = greedy_mesh(&chunk);
    assert_eq!(q.total(), 0);
}

#[test]
fn single_block_produces_six_faces() {
    let q = mesh_single(TestBlock::Stone);
    assert_eq!(q.total(), 6);
    for face in Face::ALL {
        assert_eq!(
            face_count(&q, face),
            1,
            "face {:?} should have 1 quad",
            face
        );
    }
}

#[test]
fn single_block_positions_are_unit_cube() {
    let q = mesh_single(TestBlock::Stone);

    // PosX face should be at x=1, spanning y=[0,1] and z=[0,1].
    assert_face_on_plane(&q, Face::PosX, 0, 1.0);
    let (y_min, y_max) = face_vertex_range(&q, Face::PosX, 1);
    assert!(y_min.abs() < 1e-6 && (y_max - 1.0).abs() < 1e-6);
    let (z_min, z_max) = face_vertex_range(&q, Face::PosX, 2);
    assert!(z_min.abs() < 1e-6 && (z_max - 1.0).abs() < 1e-6);

    // NegY face should be at y=0.
    assert_face_on_plane(&q, Face::NegY, 1, 0.0);
}

#[test]
fn block_at_chunk_edge_has_all_faces() {
    let q = mesh_with(&[(15, 15, 15, TestBlock::Stone)]);
    assert_eq!(q.total(), 6);
}

#[test]
fn air_does_not_cull_adjacent_face() {
    let q = mesh_single(TestBlock::Stone);
    assert_eq!(q.total(), 6);
}

#[test]
fn blocks_only_in_padding_produce_no_quads() {
    let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
    // Place stone in every padding cell but leave the real chunk empty.
    for x in 0..PADDED {
        for y in 0..PADDED {
            for z in 0..PADDED {
                let in_real = x >= PADDING
                    && x < PADDING + CHUNK_SIZE
                    && y >= PADDING
                    && y < PADDING + CHUNK_SIZE
                    && z >= PADDING
                    && z < PADDING + CHUNK_SIZE;
                if !in_real {
                    chunk.set_padded(x, y, z, TestBlock::Stone);
                }
            }
        }
    }
    let q = greedy_mesh(&chunk);
    assert_eq!(q.total(), 0);
}
