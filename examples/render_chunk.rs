//! Software-renders a meshed voxel chunk using `euc` and writes the
//! result to `examples/render_chunk.png`.
//!
//! Demonstrates per-vertex ambient occlusion and smooth lighting with
//! BFS light propagation from lamp blocks.
//!
//! Run with:
//!   cargo run --example render_chunk

use std::collections::VecDeque;
use std::path::Path;

use euc::{
    Buffer2d, CoordinateMode, CullMode as EucCullMode, DepthMode, Pipeline, Target, Texture,
    TriangleList,
};
use vek::{Mat4, Rgba, Vec2, Vec3, Vec4};
use voxmesh::*;

// Block definition

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MyBlock {
    Air,
    Cobblestone,
    CobbleSlab, // Lower half-slab (NegY, thickness 8)
    Clay,
    Glass,
    Leaves,
    SugarCane, // Cross(0), diagonal billboard
    Cobweb,    // Cross(4), stretched diagonal billboard
    Shrub,     // Cross(0), short diagonal billboard
    Ladder,    // Facade(PosX), flat face on +X side
    Rail,      // Facade(NegY), flat face on bottom
    Debug,     // WholeBlock with UV debug texture
    Cactus,    // Inset(1), horizontal faces inset by 1/16
    ChainY,    // Cross rooted on NegY, vertical chains
    ChainX,    // Cross rooted on PosX, horizontal chains along X
    ChainZ,    // Cross rooted on PosZ, horizontal chains along Z
    Lamp,      // Light-emitting block
}

impl MyBlock {
    fn is_opaque(self) -> bool {
        matches!(
            self,
            MyBlock::Cobblestone
                | MyBlock::CobbleSlab
                | MyBlock::Clay
                | MyBlock::Debug
                | MyBlock::Lamp
                | MyBlock::Cactus
        )
    }

    fn emitted_light(self) -> u8 {
        match self {
            MyBlock::Lamp => 15,
            _ => 0,
        }
    }
}

/// A block paired with a propagated light level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LitBlock {
    block: MyBlock,
    propagated_light: u8,
}

impl Block for LitBlock {
    type TransparentGroup = ();
    type Light = u8;

    fn shape(&self) -> Shape {
        match self.block {
            MyBlock::CobbleSlab => Shape::Slab(SlabInfo {
                face: AlignedFace::NegY,
                thickness: 8,
            }),
            MyBlock::SugarCane | MyBlock::Shrub => Shape::Cross(CrossInfo {
                face: AlignedFace::NegY,
                stretch: 0,
            }),
            MyBlock::Cobweb => Shape::Cross(CrossInfo {
                face: AlignedFace::NegY,
                stretch: 4,
            }),
            MyBlock::Ladder => Shape::Facade(AlignedFace::PosX),
            MyBlock::Rail => Shape::Facade(AlignedFace::NegY),
            MyBlock::Cactus => Shape::Inset(1),
            MyBlock::ChainY => Shape::Cross(CrossInfo {
                face: AlignedFace::NegY,
                stretch: 0,
            }),
            MyBlock::ChainX => Shape::Cross(CrossInfo {
                face: AlignedFace::PosX,
                stretch: 0,
            }),
            MyBlock::ChainZ => Shape::Cross(CrossInfo {
                face: AlignedFace::PosZ,
                stretch: 0,
            }),
            _ => Shape::WholeBlock,
        }
    }

    fn cull_mode(&self) -> CullMode {
        match self.block {
            MyBlock::Air => CullMode::Empty,
            MyBlock::Cobblestone
            | MyBlock::Clay
            | MyBlock::CobbleSlab
            | MyBlock::Debug
            | MyBlock::Lamp => CullMode::Opaque,
            MyBlock::Glass => CullMode::TransparentMerged(()),
            MyBlock::Cactus
            | MyBlock::SugarCane
            | MyBlock::Cobweb
            | MyBlock::Shrub
            | MyBlock::Ladder
            | MyBlock::Rail
            | MyBlock::ChainY
            | MyBlock::ChainX
            | MyBlock::ChainZ => CullMode::TransparentUnmerged,
            MyBlock::Leaves => CullMode::TransparentUnmerged,
        }
    }

    fn light(&self) -> u8 {
        self.propagated_light
    }
}

/// BFS flood-fill light propagation across the padded chunk.
fn propagate_light(chunk: &mut PaddedChunk<LitBlock>) {
    let mut queue = VecDeque::new();

    // Seed with all emitting blocks.
    for i in 0..PADDED_VOLUME {
        let b = chunk.data[i];
        if b.propagated_light > 0 {
            queue.push_back((i, b.propagated_light));
        }
    }

    let strides: [isize; 6] = [
        1,
        -1,
        PADDED as isize,
        -(PADDED as isize),
        (PADDED * PADDED) as isize,
        -((PADDED * PADDED) as isize),
    ];

    while let Some((idx, level)) = queue.pop_front() {
        if level <= 1 {
            continue;
        }
        let new_level = level - 1;

        for &stride in &strides {
            let ni = idx as isize + stride;
            if ni < 0 || ni >= PADDED_VOLUME as isize {
                continue;
            }
            let ni = ni as usize;
            let neighbor = &mut chunk.data[ni];
            // Propagate through non-opaque blocks (or any block with lower light).
            if !neighbor.block.is_opaque() && neighbor.propagated_light < new_level {
                neighbor.propagated_light = new_level;
                queue.push_back((ni, new_level));
            }
        }
    }
}

// Texture atlas

fn load_tile(path: &str) -> Vec<Rgba<f32>> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open {path}: {e}"))
        .to_rgba8();
    assert_eq!(img.width(), 16);
    assert_eq!(img.height(), 16);
    img.pixels()
        .map(|p| Rgba::new(p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32))
        .collect()
}

fn build_atlas() -> Buffer2d<Rgba<f32>> {
    let tiles: Vec<Vec<Rgba<f32>>> = [
        "examples/cobblestone.png",
        "examples/clay.png",
        "examples/glass.png",
        "examples/leaves.png",
        "examples/sugarcane.png",
        "examples/cobweb.png",
        "examples/shrub.png",
        "examples/ladder.png",
        "examples/rail_straight.png",
        "examples/debug.png",
        "examples/cactus_side.png",
        "examples/cactus_top.png",
        "examples/cactus_bottom.png",
        "examples/chain.png",
        "examples/lamp.png",
    ]
    .iter()
    .map(|p| load_tile(p))
    .collect();

    let atlas_w = tiles.len() * 16;
    let mut atlas = Buffer2d::fill([atlas_w, 16], Rgba::zero());
    for (tile_idx, tile) in tiles.iter().enumerate() {
        for ty in 0..16usize {
            for tx in 0..16usize {
                let x = tile_idx * 16 + tx;
                atlas.write(x, ty, tile[ty * 16 + tx]);
            }
        }
    }
    atlas
}

/// Returns the U offset (0..N) into the atlas strip for a given block/face.
fn atlas_u_offset(block: MyBlock, face: Face) -> f32 {
    match block {
        MyBlock::Cobblestone | MyBlock::CobbleSlab => 0.0,
        MyBlock::Clay => 1.0,
        MyBlock::Glass => 2.0,
        MyBlock::Leaves => 3.0,
        MyBlock::SugarCane => 4.0,
        MyBlock::Cobweb => 5.0,
        MyBlock::Shrub => 6.0,
        MyBlock::Ladder => 7.0,
        MyBlock::Rail => 8.0,
        MyBlock::Debug => 9.0,
        MyBlock::Cactus => match face {
            Face::Aligned(AlignedFace::PosY) => 11.0,
            Face::Aligned(AlignedFace::NegY) => 12.0,
            _ => 10.0,
        },
        MyBlock::ChainY | MyBlock::ChainX | MyBlock::ChainZ => 13.0,
        MyBlock::Lamp => 14.0,
        MyBlock::Air => 0.0,
    }
}

// Vertex type

#[derive(Clone)]
struct Vertex {
    pos: Vec4<f32>,
    uv: Vec2<f32>,
    normal: Vec3<f32>,
    /// Per-vertex AO (0.0 = fully occluded, 1.0 = fully lit).
    ao: f32,
    /// Per-vertex smooth light (0.0..1.0).
    smooth_light: f32,
    // Block that produced this quad, for render pass filtering.
    block: MyBlock,
    // Pre-computed atlas tile offset for this face.
    atlas_offset: f32,
    // Whether this vertex belongs to a two-sided quad.
    two_sided: bool,
}

// Interpolated vertex-to-fragment data.
#[derive(Clone)]
struct VsOut {
    uv: Vec2<f32>,
    normal: Vec3<f32>,
    atlas_u_offset: f32,
    ao: f32,
    smooth_light: f32,
}

impl std::ops::Mul<f32> for VsOut {
    type Output = Self;
    fn mul(self, w: f32) -> Self {
        VsOut {
            uv: self.uv * w,
            normal: self.normal * w,
            atlas_u_offset: self.atlas_u_offset * w,
            ao: self.ao * w,
            smooth_light: self.smooth_light * w,
        }
    }
}

impl std::ops::Add for VsOut {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        VsOut {
            uv: self.uv + rhs.uv,
            normal: self.normal + rhs.normal,
            atlas_u_offset: self.atlas_u_offset + rhs.atlas_u_offset,
            ao: self.ao + rhs.ao,
            smooth_light: self.smooth_light + rhs.smooth_light,
        }
    }
}

// Pipeline

struct ChunkPipeline<'a> {
    mvp: Mat4<f32>,
    atlas: &'a Buffer2d<Rgba<f32>>,
    cull_mode: EucCullMode,
}

impl<'r, 'a: 'r> Pipeline<'r> for ChunkPipeline<'a> {
    type Vertex = Vertex;
    type VertexData = VsOut;
    type Primitives = TriangleList;
    type Fragment = Rgba<f32>;
    type Pixel = u32;

    fn coordinate_mode(&self) -> CoordinateMode {
        CoordinateMode::OPENGL
    }

    fn depth_mode(&self) -> DepthMode {
        DepthMode::LESS_WRITE
    }

    fn rasterizer_config(
        &self,
    ) -> <<Self::Primitives as euc::primitives::PrimitiveKind<Self::VertexData>>::Rasterizer as euc::rasterizer::Rasterizer>::Config
    {
        self.cull_mode
    }

    fn vertex(&self, v: &Self::Vertex) -> ([f32; 4], Self::VertexData) {
        let mut clip = self.mvp * v.pos;
        // Flip Y since euc rasterizes with Y-down.
        clip.y = -clip.y;
        (
            clip.into_array(),
            VsOut {
                uv: v.uv,
                normal: v.normal,
                atlas_u_offset: v.atlas_offset,
                ao: v.ao,
                smooth_light: v.smooth_light,
            },
        )
    }

    fn fragment(&self, vs: Self::VertexData) -> Self::Fragment {
        // Tile the UV within [0,1) and look up in the atlas strip.
        let u_frac = vs.uv.x.rem_euclid(1.0);
        let v_frac = vs.uv.y.rem_euclid(1.0);

        let tile_offset = vs.atlas_u_offset.round();
        let atlas_size = self.atlas.size();
        let px = ((tile_offset * 16.0 + u_frac * 15.999) as usize).min(atlas_size[0] - 1);
        let py = ((1.0 - v_frac) * 15.999) as usize;

        let texel = self.atlas.read([px, py]);

        // Directional lighting.
        let light_dir = Vec3::new(0.4, 0.7, 0.5).normalized();
        let ndotl = vs.normal.normalized().dot(light_dir).max(0.0);
        let ambient = 0.35;
        let directional = ambient + (1.0 - ambient) * ndotl;

        // Combine: directional * AO + voxel light contribution.
        let shade = directional * vs.ao + vs.smooth_light * 0.6;

        Rgba::new(texel.r * shade, texel.g * shade, texel.b * shade, texel.a)
    }

    fn blend(&self, old: Self::Pixel, new: Self::Fragment) -> Self::Pixel {
        let alpha = (new.a * (1.0 / 255.0)).clamp(0.0, 1.0);
        if alpha < 0.01 {
            return old;
        }
        let [ob, og, or, _oa] = old.to_le_bytes();
        let r = (new.r * alpha + or as f32 * (1.0 - alpha)).clamp(0.0, 255.0) as u8;
        let g = (new.g * alpha + og as f32 * (1.0 - alpha)).clamp(0.0, 255.0) as u8;
        let b = (new.b * alpha + ob as f32 * (1.0 - alpha)).clamp(0.0, 255.0) as u8;
        u32::from_le_bytes([b, g, r, 255])
    }
}

// Scene construction

fn build_chunk() -> PaddedChunk<LitBlock> {
    use glam::UVec3;

    let air = LitBlock {
        block: MyBlock::Air,
        propagated_light: 0,
    };
    let mut chunk = PaddedChunk::new_filled(air);

    let set = |chunk: &mut PaddedChunk<LitBlock>, x: u32, y: u32, z: u32, block: MyBlock| {
        chunk.set(
            UVec3::new(x, y, z),
            LitBlock {
                block,
                propagated_light: block.emitted_light(),
            },
        );
    };

    // Cobblestone floor (y=0).
    for x in 0..CHUNK_SIZE as u32 {
        for z in 0..CHUNK_SIZE as u32 {
            set(&mut chunk, x, 0, z, MyBlock::Cobblestone);
        }
    }

    // Clay pillars at corners.
    let pillars = [(2, 2), (2, 13), (13, 2), (13, 13)];
    for &(px, pz) in &pillars {
        for y in 1..6 {
            set(&mut chunk, px, y, pz, MyBlock::Clay);
        }
    }

    // Sugar cane stalks (3 blocks tall).
    for &(sx, sz) in &[(1, 1), (1, 2), (2, 1), (14, 14), (14, 15)] {
        for y in 1..4 {
            set(&mut chunk, sx, y, sz, MyBlock::SugarCane);
        }
    }

    // Cobwebs in the upper corners between pillars.
    for &(cx, cz) in &[(3, 3), (3, 12), (12, 3), (12, 12)] {
        set(&mut chunk, cx, 5, cz, MyBlock::Cobweb);
    }

    // Debug blocks for UV visualization.
    set(&mut chunk, 7, 1, 5, MyBlock::Debug);
    set(&mut chunk, 9, 1, 5, MyBlock::Debug);

    // Shrubs scattered around.
    for &(sx, sz) in &[(1, 3), (6, 10), (10, 6), (9, 11), (4, 4)] {
        set(&mut chunk, sx, 1, sz, MyBlock::Shrub);
    }

    // Cactus pillars (3 blocks tall).
    for &(cx, cz) in &[(4, 1), (10, 10)] {
        for y in 1..4 {
            set(&mut chunk, cx, y, cz, MyBlock::Cactus);
        }
    }

    // Lamp blocks as light sources.
    set(&mut chunk, 8, 1, 5, MyBlock::Lamp);
    set(&mut chunk, 5, 5, 5, MyBlock::Lamp);
    set(&mut chunk, 10, 5, 10, MyBlock::Lamp);

    // Chains surrounding the building (all 3 cross root axes).
    for &(cx, cz) in &[
        (1, 5),
        (1, 10),
        (14, 5),
        (14, 10),
        (5, 1),
        (10, 1),
        (5, 14),
        (10, 14),
    ] {
        for y in 2..6 {
            set(&mut chunk, cx, y, cz, MyBlock::ChainY);
        }
    }
    for x in 4..12 {
        set(&mut chunk, x, 4, 1, MyBlock::ChainX);
        set(&mut chunk, x, 4, 14, MyBlock::ChainX);
    }
    for z in 4..12 {
        set(&mut chunk, 1, 4, z, MyBlock::ChainZ);
        set(&mut chunk, 14, 4, z, MyBlock::ChainZ);
    }

    // Ladders on the +X face of clay pillars.
    for y in 1..6 {
        set(&mut chunk, 3, y, 2, MyBlock::Ladder);
        set(&mut chunk, 3, y, 13, MyBlock::Ladder);
    }

    // Rails on the floor along the slab path.
    for i in 3..13 {
        set(&mut chunk, i, 1, 7, MyBlock::Rail);
    }

    // Glass windows between pillars (along edges).
    for i in 3..13 {
        for y in 1..5 {
            set(&mut chunk, 2, y, i, MyBlock::Glass);
            set(&mut chunk, 13, y, i, MyBlock::Glass);
            set(&mut chunk, i, y, 2, MyBlock::Glass);
            set(&mut chunk, i, y, 13, MyBlock::Glass);
        }
    }

    // Cobblestone half-slab path through the interior.
    for i in 3..13 {
        set(&mut chunk, i, 1, 8, MyBlock::CobbleSlab);
        set(&mut chunk, 8, 1, i, MyBlock::CobbleSlab);
    }

    // Leaves canopy on top.
    for x in 1..15 {
        for z in 1..15 {
            set(&mut chunk, x, 6, z, MyBlock::Leaves);
        }
    }

    // Propagate light from lamp blocks through air/transparent blocks.
    propagate_light(&mut chunk);

    chunk
}

/// Converts voxmesh quads into triangle vertices suitable for euc.
/// Uses per-vertex AO and smooth light from the mesher, and applies
/// the anisotropy fix for correct AO interpolation.
fn quads_to_vertices(quads: &Quads<u8>, chunk: &PaddedChunk<LitBlock>) -> Vec<Vertex> {
    let mut verts = Vec::new();

    for qf in Face::ALL {
        for quad in quads.get(qf) {
            let vp = quad.voxel_position(qf);
            let lit_block = *chunk.get_padded(vp + glam::UVec3::splat(PADDING as u32));

            let n = qf.normal(lit_block.shape());
            let normal = Vec3::new(n.x, n.y, n.z);

            let positions = quad.positions(qf, lit_block.shape());
            let uvs = quad.texture_coordinates(qf, Axis::X, false);

            let two_sided = qf.is_diagonal() || matches!(lit_block.shape(), Shape::Facade(_));
            let atlas_off = atlas_u_offset(lit_block.block, qf);

            let make_vert = |i: usize| Vertex {
                pos: Vec4::new(positions[i].x, positions[i].y, positions[i].z, 1.0),
                uv: Vec2::new(uvs[i].x, uvs[i].y),
                normal,
                ao: quad.ao[i] as f32 / 3.0,
                smooth_light: quad.light[i] as f32 / 15.0,
                block: lit_block.block,
                atlas_offset: atlas_off,
                two_sided,
            };

            // Use anisotropy-corrected indices for proper AO interpolation.
            let indices = quad.indices_ao(0);
            for &idx in &indices {
                verts.push(make_vert(idx as usize));
            }
        }
    }

    verts
}

// Main

fn main() {
    let [w, h]: [usize; 2] = [800, 600];

    let atlas = build_atlas();
    let chunk = build_chunk();
    let quads = mesh_chunk(&chunk, true);

    println!(
        "Mesh produced {} quads ({} triangles)",
        quads.total(),
        quads.total() * 2,
    );

    let vertices = quads_to_vertices(&quads, &chunk);

    // Camera looking at the chunk center from an elevated angle.
    let center = Vec3::new(8.0, 3.0, 8.0);
    let eye = Vec3::new(-6.0, 14.0, -6.0);
    let up = Vec3::new(0.0, 1.0, 0.0);

    let view = Mat4::look_at_rh(eye, center, up);
    let proj = Mat4::perspective_fov_rh_no(0.9, w as f32, h as f32, 0.1, 100.0);
    let mvp = proj * view;

    let mut color = Buffer2d::fill([w, h], 0u32);
    let mut depth = Buffer2d::fill([w, h], 1.0f32);

    // Sky gradient background.
    for y in 0..h {
        let t = y as f32 / h as f32;
        let r = (0.45 + 0.35 * t).min(1.0);
        let g = (0.60 + 0.30 * t).min(1.0);
        let b = (0.85 + 0.10 * t).min(1.0);
        let pixel =
            u32::from_le_bytes([(b * 255.0) as u8, (g * 255.0) as u8, (r * 255.0) as u8, 255]);
        for x in 0..w {
            color.write(x, y, pixel);
        }
    }

    let pipeline = ChunkPipeline {
        mvp,
        atlas: &atlas,
        cull_mode: EucCullMode::Back,
    };
    let diag_pipeline = ChunkPipeline {
        mvp,
        atlas: &atlas,
        cull_mode: EucCullMode::None,
    };

    // Render opaque geometry first.
    let opaque_verts: Vec<&Vertex> = vertices
        .iter()
        .filter(|v| {
            !v.two_sided
                && matches!(
                    v.block,
                    MyBlock::Cobblestone
                        | MyBlock::CobbleSlab
                        | MyBlock::Clay
                        | MyBlock::Debug
                        | MyBlock::Cactus
                        | MyBlock::Lamp
                )
        })
        .collect();
    if !opaque_verts.is_empty() {
        pipeline.render(opaque_verts.iter().map(|v| *v), &mut color, &mut depth);
    }

    // Two-sided geometry (cross-shaped billboards, facades), no backface culling.
    let two_sided_verts: Vec<&Vertex> = vertices.iter().filter(|v| v.two_sided).collect();
    if !two_sided_verts.is_empty() {
        diag_pipeline.render(two_sided_verts.iter().map(|v| *v), &mut color, &mut depth);
    }

    // Then transparent geometry (glass, leaves).
    let transparent_verts: Vec<&Vertex> = vertices
        .iter()
        .filter(|v| !v.two_sided && matches!(v.block, MyBlock::Glass | MyBlock::Leaves))
        .collect();
    if !transparent_verts.is_empty() {
        pipeline.render(transparent_verts.iter().map(|v| *v), &mut color, &mut depth);
    }

    // Write output.
    let out_path = Path::new("examples/render_chunk.png");
    let mut img = image::RgbaImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let pixel = color.read([x, y]);
            let [b, g, r, a] = pixel.to_le_bytes();
            img.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, a]));
        }
    }
    img.save(out_path).expect("failed to write output image");
    println!("Wrote {}", out_path.display());
}
