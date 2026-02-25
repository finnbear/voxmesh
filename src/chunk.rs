use glam::UVec3;

use crate::block::Block;

pub const CHUNK_SIZE: usize = 16;
pub const PADDING: usize = 1;
pub const PADDED: usize = CHUNK_SIZE + 2 * PADDING; // 18
pub const PADDED_VOLUME: usize = PADDED * PADDED * PADDED;

#[inline]
pub fn linearize(padded_pos: UVec3) -> usize {
    padded_pos.x as usize + padded_pos.y as usize * PADDED + padded_pos.z as usize * PADDED * PADDED
}

#[inline]
pub fn delinearize(index: usize) -> UVec3 {
    let i = index as u32;
    const P: u32 = PADDING as u32;
    UVec3::new(i % P, (i / P) % P, i / (P * P))
}

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
