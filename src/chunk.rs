use crate::block::Block;

pub const CHUNK_SIZE: usize = 16;
pub const PADDING: usize = 1;
pub const PADDED: usize = CHUNK_SIZE + 2 * PADDING; // 18
pub const PADDED_VOLUME: usize = PADDED * PADDED * PADDED;

#[inline]
pub fn padded_idx(x: usize, y: usize, z: usize) -> usize {
    x + y * PADDED + z * PADDED * PADDED
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
    pub fn set(&mut self, x: usize, y: usize, z: usize, block: B) {
        self.data[padded_idx(x + PADDING, y + PADDING, z + PADDING)] = block;
    }

    #[inline]
    pub fn set_padded(&mut self, x: usize, y: usize, z: usize, block: B) {
        self.data[padded_idx(x, y, z)] = block;
    }

    #[inline]
    pub fn get_padded(&self, x: usize, y: usize, z: usize) -> &B {
        &self.data[padded_idx(x, y, z)]
    }
}
