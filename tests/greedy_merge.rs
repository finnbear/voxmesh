mod common;

use common::*;
use voxmesh::*;

#[test]
fn flat_layer_merges_into_one_quad_per_face() {
    let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
    for x in 0..CHUNK_SIZE as u32 {
        for z in 0..CHUNK_SIZE as u32 {
            chunk.set(glam::UVec3::new(x, 0, z), TestBlock::Stone);
        }
    }
    let q = greedy_mesh(&chunk);
    for face in Face::ALL {
        assert_eq!(face_count(&q, face), 1, "face {:?}", face);
    }
    assert_eq!(q.total(), 6);
}

#[test]
fn flat_slab_layer_merges_into_one_quad_per_face() {
    let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
    for x in 0..CHUNK_SIZE as u32 {
        for z in 0..CHUNK_SIZE as u32 {
            chunk.set(glam::UVec3::new(x, 0, z), TestBlock::LowerSlab);
        }
    }
    let q = greedy_mesh(&chunk);
    for face in Face::ALL {
        assert_eq!(face_count(&q, face), 1, "face {:?}", face);
    }
    assert_eq!(q.total(), 6);
}

#[test]
fn full_chunk_same_block_produces_six_quads() {
    let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
    for x in 0..CHUNK_SIZE as u32 {
        for y in 0..CHUNK_SIZE as u32 {
            for z in 0..CHUNK_SIZE as u32 {
                chunk.set(glam::UVec3::new(x, y, z), TestBlock::Stone);
            }
        }
    }
    let q = greedy_mesh(&chunk);
    assert_eq!(q.total(), 6);
    for face in Face::ALL {
        assert_eq!(face_count(&q, face), 1, "face {:?}", face);
    }
}

#[test]
fn checkerboard_produces_many_quads() {
    let mut chunk = PaddedChunk::new_filled(TestBlock::Air);
    for x in 0..CHUNK_SIZE as u32 {
        for z in 0..CHUNK_SIZE as u32 {
            if (x + z) % 2 == 0 {
                chunk.set(glam::UVec3::new(x, 0, z), TestBlock::Stone);
            }
        }
    }
    let q = greedy_mesh(&chunk);
    let block_count = (CHUNK_SIZE * CHUNK_SIZE + 1) / 2; // 128
    assert_eq!(q.total(), block_count * 6);
}
