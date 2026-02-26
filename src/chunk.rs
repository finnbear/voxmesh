use glam::UVec3;

use crate::block::Block;

/// Side length of the inner chunk in blocks.
pub const CHUNK_SIZE: usize = 16;
/// Width of the padding ring around the chunk (1 block on each side).
pub const PADDING: usize = 1;
/// Side length of the padded chunk (`CHUNK_SIZE + 2 * PADDING`).
pub const PADDED: usize = CHUNK_SIZE + 2 * PADDING; // 18
/// Total number of voxels in the padded chunk (`PADDED³`).
pub const PADDED_VOLUME: usize = PADDED * PADDED * PADDED;

/// Converts a padded 3D position to a linear index into the chunk data array.
#[inline]
pub fn linearize(padded_pos: UVec3) -> usize {
    padded_pos.x as usize + padded_pos.y as usize * PADDED + padded_pos.z as usize * PADDED * PADDED
}

/// Converts a linear index back to a padded 3D position.
#[inline]
pub fn delinearize(index: usize) -> UVec3 {
    let i = index as u32;
    const P: u32 = PADDED as u32;
    UVec3::new(i % P, (i / P) % P, i / (P * P))
}

/// A 16³ chunk of blocks with a 1-block padding ring on all sides.
#[derive(Clone)]
pub struct PaddedChunk<B: Block> {
    pub data: [B; PADDED_VOLUME],
}

impl<B: Block> PaddedChunk<B> {
    pub fn new_filled(fill: B) -> Self {
        PaddedChunk {
            data: [fill; PADDED_VOLUME],
        }
    }

    #[inline]
    pub fn set(&mut self, pos: UVec3, block: B) {
        let p = PADDING as u32;
        self.data[linearize(pos + UVec3::splat(p))] = block;
    }

    #[inline]
    pub fn set_padded(&mut self, pos: UVec3, block: B) {
        self.data[linearize(pos)] = block;
    }

    #[inline]
    pub fn get_padded(&self, pos: UVec3) -> &B {
        &self.data[linearize(pos)]
    }
}
