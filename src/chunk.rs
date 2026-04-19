use std::marker::PhantomData;

use glam::UVec3;

use crate::block::Block;

/// Width of the padding ring around the chunk (1 block on each side).
pub const PADDING: usize = 1;

/// Describes the dimensions of a cubic chunk.
///
/// Implementors need only specify [`SIZE`](ChunkShape::SIZE); the
/// padded size, volume, and indexing functions are derived automatically.
pub trait ChunkShape {
    /// Side length of the inner chunk in blocks (e.g. 16).
    const SIZE: usize;
    /// Side length of the padded chunk (`SIZE + 2 * PADDING`).
    const PADDED: usize = Self::SIZE + 2 * PADDING;
    /// Total number of voxels in the padded chunk (`PADDED³`).
    const PADDED_VOLUME: usize = Self::PADDED * Self::PADDED * Self::PADDED;

    /// Converts a padded 3D position to a linear index.
    #[inline]
    fn linearize(padded_pos: UVec3) -> usize {
        padded_pos.x as usize
            + padded_pos.y as usize * Self::PADDED
            + padded_pos.z as usize * Self::PADDED * Self::PADDED
    }

    /// Converts a linear index back to a padded 3D position.
    #[inline]
    fn delinearize(index: usize) -> UVec3 {
        let p = Self::PADDED;
        UVec3::new(
            (index % p) as u32,
            ((index / p) % p) as u32,
            (index / (p * p)) as u32,
        )
    }
}

/// A 16-wide chunk shape (default).
pub struct ChunkShape16;
impl ChunkShape for ChunkShape16 {
    const SIZE: usize = 16;
}

/// An 8-wide chunk shape.
pub struct ChunkShape8;
impl ChunkShape for ChunkShape8 {
    const SIZE: usize = 8;
}

/// A 4-wide chunk shape.
pub struct ChunkShape4;
impl ChunkShape for ChunkShape4 {
    const SIZE: usize = 4;
}

/// A 2-wide chunk shape.
pub struct ChunkShape2;
impl ChunkShape for ChunkShape2 {
    const SIZE: usize = 2;
}

/// A chunk of blocks with a 1-block padding ring on all sides.
///
/// The shape `S` determines the inner width. Use the
/// [`PaddedChunk16`] type alias for the common 16-wide case.
#[derive(Clone)]
pub struct PaddedChunk<B: Block, S: ChunkShape>
where
    [(); S::PADDED_VOLUME]:,
{
    pub data: [B; S::PADDED_VOLUME],
    _shape: PhantomData<S>,
}

/// A [`PaddedChunk`] using the default 16-wide [`ChunkShape16`].
pub type PaddedChunk16<B> = PaddedChunk<B, ChunkShape16>;

impl<B: Block, S: ChunkShape> PaddedChunk<B, S>
where
    [(); S::PADDED_VOLUME]:,
{
    pub fn new_filled(fill: B) -> Self {
        PaddedChunk {
            data: [fill; S::PADDED_VOLUME],
            _shape: PhantomData,
        }
    }

    #[inline]
    pub fn set(&mut self, pos: UVec3, block: B) {
        let p = PADDING as u32;
        self.data[S::linearize(pos + UVec3::splat(p))] = block;
    }

    #[inline]
    pub fn set_padded(&mut self, pos: UVec3, block: B) {
        self.data[S::linearize(pos)] = block;
    }

    #[inline]
    pub fn get_padded(&self, pos: UVec3) -> &B {
        &self.data[S::linearize(pos)]
    }
}
