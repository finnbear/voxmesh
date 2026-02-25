#![feature(test)]

extern crate test;

use test::Bencher;

use block_mesh::{
    greedy_quads, GreedyQuadsBuffer, MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};
use ndshape::{ConstShape, ConstShape3u32};
use voxmesh::{greedy_mesh_into, *};

// ---------------------------------------------------------------------------
// voxmesh block type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VmBlock {
    Air,
    Stone,
}

impl Block for VmBlock {
    type TransparentGroup = ();

    fn shape(&self) -> Shape {
        Shape::WholeBlock
    }
    fn cull_mode(&self) -> CullMode {
        match self {
            VmBlock::Air => CullMode::Empty,
            VmBlock::Stone => CullMode::Opaque,
        }
    }
}

// ---------------------------------------------------------------------------
// block-mesh voxel type
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BmVoxel {
    Air,
    Stone,
}

impl Voxel for BmVoxel {
    fn get_visibility(&self) -> VoxelVisibility {
        match self {
            BmVoxel::Air => VoxelVisibility::Empty,
            BmVoxel::Stone => VoxelVisibility::Opaque,
        }
    }
}

impl MergeVoxel for BmVoxel {
    type MergeValue = Self;
    fn merge_value(&self) -> Self::MergeValue {
        *self
    }
}

type BmPaddedShape = ConstShape3u32<18, 18, 18>;

// ---------------------------------------------------------------------------
// Helpers to fill chunks with the same pattern for both libraries
// ---------------------------------------------------------------------------

/// Solid 16^3 chunk — worst case for face count (only exterior faces survive).
fn vm_chunk_solid() -> PaddedChunk<VmBlock> {
    let mut chunk = PaddedChunk::new_filled(VmBlock::Air);
    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                chunk.set(x, y, z, VmBlock::Stone);
            }
        }
    }
    chunk
}

fn bm_chunk_solid() -> [BmVoxel; BmPaddedShape::SIZE as usize] {
    let mut voxels = [BmVoxel::Air; BmPaddedShape::SIZE as usize];
    for x in 1u32..17 {
        for y in 1u32..17 {
            for z in 1u32..17 {
                voxels[BmPaddedShape::linearize([x, y, z]) as usize] = BmVoxel::Stone;
            }
        }
    }
    voxels
}

/// Checkerboard — worst case for merging (no two adjacent same-type blocks).
fn vm_chunk_checkerboard() -> PaddedChunk<VmBlock> {
    let mut chunk = PaddedChunk::new_filled(VmBlock::Air);
    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                if (x + y + z) % 2 == 0 {
                    chunk.set(x, y, z, VmBlock::Stone);
                }
            }
        }
    }
    chunk
}

fn bm_chunk_checkerboard() -> [BmVoxel; BmPaddedShape::SIZE as usize] {
    let mut voxels = [BmVoxel::Air; BmPaddedShape::SIZE as usize];
    for x in 0u32..16 {
        for y in 0u32..16 {
            for z in 0u32..16 {
                if (x + y + z) % 2 == 0 {
                    voxels[BmPaddedShape::linearize([x + 1, y + 1, z + 1]) as usize] =
                        BmVoxel::Stone;
                }
            }
        }
    }
    voxels
}

/// Hollow shell — floor + walls + ceiling, interior is air.
fn vm_chunk_shell() -> PaddedChunk<VmBlock> {
    let mut chunk = PaddedChunk::new_filled(VmBlock::Air);
    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                if x == 0 || x == 15 || y == 0 || y == 15 || z == 0 || z == 15 {
                    chunk.set(x, y, z, VmBlock::Stone);
                }
            }
        }
    }
    chunk
}

fn bm_chunk_shell() -> [BmVoxel; BmPaddedShape::SIZE as usize] {
    let mut voxels = [BmVoxel::Air; BmPaddedShape::SIZE as usize];
    for x in 0u32..16 {
        for y in 0u32..16 {
            for z in 0u32..16 {
                if x == 0 || x == 15 || y == 0 || y == 15 || z == 0 || z == 15 {
                    voxels[BmPaddedShape::linearize([x + 1, y + 1, z + 1]) as usize] =
                        BmVoxel::Stone;
                }
            }
        }
    }
    voxels
}

// ---------------------------------------------------------------------------
// block-mesh runner (reuses buffer across iterations)
// ---------------------------------------------------------------------------

fn bench_block_mesh(b: &mut Bencher, voxels: &[BmVoxel; BmPaddedShape::SIZE as usize]) {
    let faces = &RIGHT_HANDED_Y_UP_CONFIG.faces;
    let mut buffer = GreedyQuadsBuffer::new(voxels.len());
    b.iter(|| {
        buffer.reset(voxels.len());
        greedy_quads(
            voxels,
            &BmPaddedShape {},
            [0; 3],
            [17; 3],
            faces,
            &mut buffer,
        );
        test::black_box(&buffer);
    });
}

// ---------------------------------------------------------------------------
// voxmesh benches
// ---------------------------------------------------------------------------

#[bench]
fn voxmesh_solid(b: &mut Bencher) {
    let chunk = vm_chunk_solid();
    let mut quads = Quads::new();
    b.iter(|| {
        greedy_mesh_into(test::black_box(&chunk), &mut quads);
        test::black_box(&quads);
    });
}

#[bench]
fn voxmesh_checkerboard(b: &mut Bencher) {
    let chunk = vm_chunk_checkerboard();
    let mut quads = Quads::new();
    b.iter(|| {
        greedy_mesh_into(test::black_box(&chunk), &mut quads);
        test::black_box(&quads);
    });
}

#[bench]
fn voxmesh_shell(b: &mut Bencher) {
    let chunk = vm_chunk_shell();
    let mut quads = Quads::new();
    b.iter(|| {
        greedy_mesh_into(test::black_box(&chunk), &mut quads);
        test::black_box(&quads);
    });
}

// ---------------------------------------------------------------------------
// block-mesh benches
// ---------------------------------------------------------------------------

#[bench]
fn block_mesh_solid(b: &mut Bencher) {
    let voxels = bm_chunk_solid();
    bench_block_mesh(b, &voxels);
}

#[bench]
fn block_mesh_checkerboard(b: &mut Bencher) {
    let voxels = bm_chunk_checkerboard();
    bench_block_mesh(b, &voxels);
}

#[bench]
fn block_mesh_shell(b: &mut Bencher) {
    let voxels = bm_chunk_shell();
    bench_block_mesh(b, &voxels);
}
