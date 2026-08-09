#![feature(associated_type_defaults)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod block;
mod chunk;
mod face;
mod light;
mod mesh;

pub use block::{
    Block, CrossInfo, CullMode, FacadeInfo, FluidInfo, Shape, SlabInfo, Thickness, FULL_THICKNESS,
};
pub use chunk::{
    ChunkShape, ChunkShape16, ChunkShape2, ChunkShape4, ChunkShape8, PaddedChunk, PaddedChunk16,
    PADDING,
};
pub use face::{AlignedFace, Axis, DiagonalFace, Face};
pub use light::Light;
pub use mesh::{mesh_block, mesh_block_into, mesh_chunk, mesh_chunk_into, Quad, Quads};
