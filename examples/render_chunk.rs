//! Software-renders a greedy-meshed voxel chunk using `euc` and writes the
//! result to `examples/render_chunk.png`.
//!
//! The chunk contains five block types arranged in a small scene:
//! - Cobblestone floor
//! - Cobblestone half-slabs (paths/steps)
//! - Clay pillars
//! - Glass windows
//! - Leaves canopy
//!
//! Run with:
//!   cargo run --example render_chunk

use std::path::Path;

use euc::{
    Buffer2d, CoordinateMode, CullMode as EucCullMode, DepthMode, Pipeline, Target, Texture,
    TriangleList,
};
use vek::{Mat4, Rgba, Vec2, Vec3, Vec4};
use voxmesh::*;

// ---------------------------------------------------------------------------
// Block definition
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MyBlock {
    Air,
    Cobblestone,
    CobbleSlab, // Lower half-slab (NegY, thickness 8)
    Clay,
    Glass,
    Leaves,
    SugarCane, // Cross(0) — diagonal billboard
    Cobweb,    // Cross(4) — stretched diagonal billboard
}

impl Block for MyBlock {
    fn shape(&self) -> Shape {
        match self {
            MyBlock::CobbleSlab => Shape::Slab(SlabInfo {
                face: Face::NegY,
                thickness: 8,
            }),
            MyBlock::SugarCane => Shape::Cross(0),
            MyBlock::Cobweb => Shape::Cross(4),
            _ => Shape::WholeBlock,
        }
    }

    fn cull_mode(&self) -> CullMode {
        match self {
            MyBlock::Air => CullMode::Empty,
            MyBlock::Cobblestone | MyBlock::Clay | MyBlock::CobbleSlab => CullMode::Opaque,
            MyBlock::Glass => CullMode::TransparentMerged,
            MyBlock::Leaves | MyBlock::SugarCane | MyBlock::Cobweb => {
                CullMode::TransparentUnmerged
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Texture atlas — pack four 16×16 tiles into a 64×16 strip
// ---------------------------------------------------------------------------

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

/// Returns the U offset (0..4) into the atlas strip for a given block.
fn atlas_u_offset(block: MyBlock) -> f32 {
    match block {
        MyBlock::Cobblestone | MyBlock::CobbleSlab => 0.0,
        MyBlock::Clay => 1.0,
        MyBlock::Glass => 2.0,
        MyBlock::Leaves => 3.0,
        MyBlock::SugarCane => 4.0,
        MyBlock::Cobweb => 5.0,
        MyBlock::Air => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Vertex type
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Vertex {
    pos: Vec4<f32>,
    uv: Vec2<f32>,
    normal: Vec3<f32>,
    // Which block produced this quad (used for atlas lookup).
    block: MyBlock,
}

// Interpolated data passed from vertex to fragment shader.
#[derive(Clone)]
struct VsOut {
    uv: Vec2<f32>,
    normal: Vec3<f32>,
    atlas_u_offset: f32,
}

impl std::ops::Mul<f32> for VsOut {
    type Output = Self;
    fn mul(self, w: f32) -> Self {
        VsOut {
            uv: self.uv * w,
            normal: self.normal * w,
            atlas_u_offset: self.atlas_u_offset * w,
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
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

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
        // Flip Y so the image isn't upside-down (euc rasterises with Y-down).
        clip.y = -clip.y;
        (
            clip.into_array(),
            VsOut {
                uv: v.uv,
                normal: v.normal,
                atlas_u_offset: atlas_u_offset(v.block),
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
        let py = (v_frac * 15.999) as usize;

        let texel = self.atlas.read([px, py]);

        // Simple directional lighting.
        let light_dir = Vec3::new(0.4, 0.7, 0.5).normalized();
        let ndotl = vs.normal.normalized().dot(light_dir).max(0.0);
        let ambient = 0.35;
        let shade = ambient + (1.0 - ambient) * ndotl;

        Rgba::new(texel.r * shade, texel.g * shade, texel.b * shade, texel.a)
    }

    fn blend(&self, old: Self::Pixel, new: Self::Fragment) -> Self::Pixel {
        let alpha = (new.a / 255.0).clamp(0.0, 1.0);
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

// ---------------------------------------------------------------------------
// Scene construction
// ---------------------------------------------------------------------------

fn build_chunk() -> PaddedChunk<MyBlock> {
    let mut chunk = PaddedChunk::new_filled(MyBlock::Air);

    // Cobblestone floor (y=0).
    for x in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            chunk.set(x, 0, z, MyBlock::Cobblestone);
        }
    }

    // Clay pillars at corners.
    let pillars = [(2, 2), (2, 13), (13, 2), (13, 13)];
    for &(px, pz) in &pillars {
        for y in 1..6 {
            chunk.set(px, y, pz, MyBlock::CobbleSlab);
        }
    }

    // Glass windows between pillars (along edges).
    for i in 3..13 {
        for y in 1..5 {
            chunk.set(2, y, i, MyBlock::Glass);
            chunk.set(13, y, i, MyBlock::Glass);
            chunk.set(i, y, 2, MyBlock::Glass);
            chunk.set(i, y, 13, MyBlock::Glass);
        }
    }

    // Cobblestone half-slab path through the interior.
    for i in 3..13 {
        chunk.set(i, 1, 8, MyBlock::CobbleSlab);
        chunk.set(8, 1, i, MyBlock::CobbleSlab);
    }

    // Leaves canopy on top.
    for x in 1..15 {
        for z in 1..15 {
            chunk.set(x, 6, z, MyBlock::Leaves);
        }
    }

    // Sugar cane stalks (3 blocks tall).
    for &(sx, sz) in &[(5, 5), (5, 6), (6, 5), (10, 10), (10, 11)] {
        for y in 1..4 {
            chunk.set(sx, y, sz, MyBlock::SugarCane);
        }
    }

    // Cobwebs in the upper corners between pillars.
    for &(cx, cz) in &[(3, 3), (3, 12), (12, 3), (12, 12)] {
        chunk.set(cx, 5, cz, MyBlock::Cobweb);
    }

    chunk
}

/// Convert axis-aligned voxmesh quads into triangle vertices suitable for euc.
fn face_quads_to_vertices(quads: &Quads, chunk: &PaddedChunk<MyBlock>) -> Vec<Vertex> {
    let mut verts = Vec::new();

    for face in Face::ALL {
        let normal_glam = face.normal();
        let normal = Vec3::new(
            normal_glam.x as f32,
            normal_glam.y as f32,
            normal_glam.z as f32,
        );

        for quad in &quads.faces[face.index()] {
            let positions = quad.positions(face);
            let uvs = quad.texture_coordinates(face, Axis::X, false);

            // Determine which block this quad belongs to by sampling the chunk
            // at the quad's interior. We use the first vertex position shifted
            // inward by half the normal.
            let p = positions[0];
            let sample_x = (p.x - 0.5 * normal.x).floor().max(0.0) as usize;
            let sample_y = (p.y - 0.5 * normal.y).floor().max(0.0) as usize;
            let sample_z = (p.z - 0.5 * normal.z).floor().max(0.0) as usize;
            let block = *chunk.get_padded(
                (sample_x + PADDING).min(PADDED - 1),
                (sample_y + PADDING).min(PADDED - 1),
                (sample_z + PADDING).min(PADDED - 1),
            );

            // Two triangles per quad: (0,1,2) and (0,2,3).
            let make_vert = |i: usize| Vertex {
                pos: Vec4::new(positions[i].x, positions[i].y, positions[i].z, 1.0),
                uv: Vec2::new(uvs[i].x, uvs[i].y),
                normal,
                block,
            };

            verts.push(make_vert(0));
            verts.push(make_vert(1));
            verts.push(make_vert(2));

            verts.push(make_vert(0));
            verts.push(make_vert(2));
            verts.push(make_vert(3));
        }
    }

    verts
}

/// Convert diagonal voxmesh quads into triangle vertices suitable for euc.
/// These are rendered two-sided (no backface culling).
fn diagonal_quads_to_vertices(quads: &Quads, chunk: &PaddedChunk<MyBlock>) -> Vec<Vertex> {
    let mut verts = Vec::new();

    for diag in DiagonalFace::ALL {
        for quad in &quads.diagonals[diag.index()] {
            // Sample the chunk at the quad's center to determine block type
            // and stretch. Use stretch=0 positions to find the block center.
            let positions_0 = quad.diagonal_positions(diag, 0);
            let center_x = (positions_0[0].x + positions_0[1].x) * 0.5;
            let center_z = (positions_0[0].z + positions_0[1].z) * 0.5;
            let base_y = positions_0[0].y;
            let sample_x = center_x.floor().max(0.0) as usize;
            let sample_y = base_y.floor().max(0.0) as usize;
            let sample_z = center_z.floor().max(0.0) as usize;
            let block = *chunk.get_padded(
                (sample_x + PADDING).min(PADDED - 1),
                (sample_y + PADDING).min(PADDED - 1),
                (sample_z + PADDING).min(PADDED - 1),
            );

            let stretch = match block.shape() {
                Shape::Cross(s) => s,
                _ => 0,
            };

            let positions = quad.diagonal_positions(diag, stretch);
            let uvs = quad.diagonal_texture_coordinates();

            // Compute a face normal from the quad vertices.
            let e1 = Vec3::new(
                positions[1].x - positions[0].x,
                positions[1].y - positions[0].y,
                positions[1].z - positions[0].z,
            );
            let e2 = Vec3::new(
                positions[3].x - positions[0].x,
                positions[3].y - positions[0].y,
                positions[3].z - positions[0].z,
            );
            let normal = Vec3::cross(e1, e2).normalized();

            let make_vert = |i: usize| Vertex {
                pos: Vec4::new(positions[i].x, positions[i].y, positions[i].z, 1.0),
                uv: Vec2::new(uvs[i].x, uvs[i].y),
                normal,
                block,
            };

            verts.push(make_vert(0));
            verts.push(make_vert(1));
            verts.push(make_vert(2));

            verts.push(make_vert(0));
            verts.push(make_vert(2));
            verts.push(make_vert(3));
        }
    }

    verts
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let [w, h]: [usize; 2] = [800, 600];

    let atlas = build_atlas();
    let chunk = build_chunk();
    let quads = greedy_mesh(&chunk);

    println!(
        "Greedy mesh produced {} quads ({} triangles)",
        quads.total(),
        quads.total() * 2,
    );

    let face_vertices = face_quads_to_vertices(&quads, &chunk);
    let diag_vertices = diagonal_quads_to_vertices(&quads, &chunk);

    // Camera: look at the chunk center from an elevated angle.
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

    // Render opaque geometry first.
    let opaque_verts: Vec<&Vertex> = face_vertices
        .iter()
        .filter(|v| {
            matches!(
                v.block,
                MyBlock::Cobblestone | MyBlock::CobbleSlab | MyBlock::Clay
            )
        })
        .collect();
    if !opaque_verts.is_empty() {
        pipeline.render(opaque_verts.iter().map(|v| *v), &mut color, &mut depth);
    }

    // Diagonal (cross-shaped) geometry — rendered two-sided.
    if !diag_vertices.is_empty() {
        let diag_pipeline = ChunkPipeline {
            mvp,
            atlas: &atlas,
            cull_mode: EucCullMode::None,
        };
        let diag_refs: Vec<&Vertex> = diag_vertices.iter().collect();
        diag_pipeline.render(diag_refs.iter().map(|v| *v), &mut color, &mut depth);
    }

    // Then transparent geometry (glass, leaves).
    let transparent_verts: Vec<&Vertex> = face_vertices
        .iter()
        .filter(|v| matches!(v.block, MyBlock::Glass | MyBlock::Leaves))
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
