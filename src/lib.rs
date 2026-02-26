#![feature(associated_type_defaults)]

mod block;
mod chunk;
mod face;
mod light;
mod mesh;

pub use block::{Block, CrossInfo, CullMode, Shape, SlabInfo};
pub use chunk::{delinearize, linearize, PaddedChunk, CHUNK_SIZE, PADDED, PADDED_VOLUME, PADDING};
pub use face::{AlignedFace, Axis, DiagonalFace, Face};
pub use light::Light;
pub use mesh::{mesh_block, mesh_block_into, mesh_chunk, mesh_chunk_into, Quad, Quads};
