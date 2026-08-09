//! Fluid surface meshing: height fields, shared vertices, and what
//! greedy merging is and is not allowed to do to them.

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use glam::UVec3;
use std::collections::HashMap;
use voxmesh::*;

/// Water strengths, as voxmaxa uses them: `0` is a source and each step
/// out from it is one shallower, down to [`MAX_STRENGTH`].
const MAX_STRENGTH: u8 = 3;

/// The height a column of each strength stands at, in 1/16ths.
const HEIGHTS: [Thickness; 4] = [16, 12, 8, 4];

const WATER: u8 = 0;
const LAVA: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FluidBlock {
    Air,
    Stone,
    Water(u8),
    Lava,
}

impl Block for FluidBlock {
    type TransparentGroup = u8;

    const FLUID_ENABLED: bool = true;

    fn shape(&self) -> Shape {
        match self {
            FluidBlock::Water(strength) => Shape::Fluid(FluidInfo {
                face: AlignedFace::PosY,
                height: HEIGHTS[(*strength).min(MAX_STRENGTH) as usize],
                id: WATER,
            }),
            FluidBlock::Lava => Shape::Fluid(FluidInfo {
                face: AlignedFace::PosY,
                height: FULL_THICKNESS,
                id: LAVA,
            }),
            _ => Shape::WholeBlock,
        }
    }

    fn cull_mode(&self) -> CullMode<u8> {
        match self {
            FluidBlock::Air => CullMode::Empty,
            FluidBlock::Stone => CullMode::Opaque,
            FluidBlock::Water(_) => CullMode::TransparentMerged(WATER),
            FluidBlock::Lava => CullMode::TransparentMerged(LAVA),
        }
    }
}

/// A meshed chunk that remembers what it was built from, so a test can
/// tell a fluid quad from the stone the fluid is standing on.
struct Scene {
    quads: Quads,
    blocks: HashMap<(u32, u32, u32), FluidBlock>,
}

impl Scene {
    fn new(blocks: &[(u32, u32, u32, FluidBlock)], greedy: bool) -> Self {
        let mut chunk = PaddedChunk16::new_filled(FluidBlock::Air);
        for &(x, y, z, block) in blocks {
            chunk.set(UVec3::new(x, y, z), block);
        }
        Scene {
            quads: mesh_chunk(&chunk, greedy),
            blocks: blocks
                .iter()
                .map(|&(x, y, z, block)| ((x, y, z), block))
                .collect(),
        }
    }

    fn is_fluid(&self, position: UVec3) -> bool {
        matches!(
            self.blocks.get(&(position.x, position.y, position.z)),
            Some(FluidBlock::Water(_) | FluidBlock::Lava),
        )
    }

    /// Every quad on `face` that a fluid produced.
    fn fluid_quads(&self, face: AlignedFace) -> impl Iterator<Item = &Quad> {
        self.quads.faces[face.index()]
            .iter()
            .filter(move |quad| self.is_fluid(quad.voxel_position(face)))
    }

    fn quad_at(&self, face: AlignedFace, position: UVec3) -> &Quad {
        self.fluid_quads(face)
            .find(|quad| quad.voxel_position(face) == position)
            .unwrap_or_else(|| panic!("no {face:?} quad at {position}"))
    }
}

fn mesh_with(blocks: &[(u32, u32, u32, FluidBlock)], greedy: bool) -> Scene {
    Scene::new(blocks, greedy)
}

/// The vertices of `quad` as `(x, z, y)` with the horizontal components
/// snapped to the 1/16th grid, sorted so two quads can be compared
/// without caring about winding.
fn corners(quad: &Quad, face: AlignedFace) -> Vec<(i32, i32, f32)> {
    let mut out: Vec<_> = quad
        .positions(
            face,
            Shape::Fluid(FluidInfo {
                face: AlignedFace::PosY,
                height: FULL_THICKNESS,
                id: WATER,
            }),
        )
        .iter()
        .map(|p| {
            (
                (p.x * 16.0).round() as i32,
                (p.z * 16.0).round() as i32,
                p.y,
            )
        })
        .collect();
    out.sort_by_key(|&(x, z, y)| (x, z, (y * 16.0).round() as i32));
    out
}

/// The surface height of every fluid top quad, keyed by its `(x, z)`
/// corner. Panics if two quads disagree about a corner they share.
fn surface_map(scene: &Scene) -> HashMap<(i32, i32), f32> {
    let mut map: HashMap<(i32, i32), f32> = HashMap::new();
    for quad in scene.fluid_quads(AlignedFace::PosY) {
        for (x, z, y) in corners(quad, AlignedFace::PosY) {
            if let Some(&existing) = map.get(&(x, z)) {
                assert!(
                    (existing - y).abs() < 1e-6,
                    "corner ({x}, {z}) is at y={existing} for one cell and y={y} for another",
                );
            }
            map.insert((x, z), y);
        }
    }
    map
}

/// A flat floor of stone at y=1 across the middle of the chunk, so a
/// fluid laid at y=2 is standing on something.
fn floor() -> Vec<(u32, u32, u32, FluidBlock)> {
    let mut blocks = Vec::new();
    for x in 0..12 {
        for z in 0..12 {
            blocks.push((x, 1, z, FluidBlock::Stone));
        }
    }
    blocks
}

#[test]
fn lone_source_is_a_whole_cube() {
    let scene = mesh_with(&[(4, 4, 4, FluidBlock::Water(0))], true);

    assert_eq!(
        scene.quads.total(),
        6,
        "a source in open air shows all six faces",
    );
    for face in AlignedFace::ALL {
        let quad = scene.quad_at(face, UVec3::new(4, 4, 4));
        assert_eq!(
            quad.corner_offsets, [0; 4],
            "face {face:?} of a lone source should sit on the block boundary",
        );
    }
}

#[test]
fn source_top_stays_full_beside_shallower_water() {
    let mut blocks = floor();
    blocks.extend([
        (4, 2, 4, FluidBlock::Water(0)),
        (5, 2, 4, FluidBlock::Water(1)),
        (3, 2, 4, FluidBlock::Water(1)),
        (4, 2, 5, FluidBlock::Water(1)),
        (4, 2, 3, FluidBlock::Water(1)),
    ]);
    let scene = mesh_with(&blocks, true);
    let source = scene.quad_at(AlignedFace::PosY, UVec3::new(4, 2, 4));

    for (x, z, y) in corners(source, AlignedFace::PosY) {
        assert!(
            (y - 3.0).abs() < 1e-6,
            "source corner ({x}, {z}) should be at the top of its block, got y={y}",
        );
    }
}

#[test]
fn flow_ramps_down_from_the_source() {
    let mut blocks = floor();
    blocks.extend([
        (4, 2, 4, FluidBlock::Water(0)),
        (5, 2, 4, FluidBlock::Water(1)),
        (6, 2, 4, FluidBlock::Water(2)),
        (7, 2, 4, FluidBlock::Water(3)),
    ]);
    let surface = surface_map(&mesh_with(&blocks, true));

    // Each corner takes the tallest column it touches, so the surface
    // steps down one strength per cell along the run and the far edge of
    // the last cell is the lowest point.
    let expected = [
        (4 * 16, 2.0 + 16.0 / 16.0),
        (5 * 16, 2.0 + 16.0 / 16.0),
        (6 * 16, 2.0 + 12.0 / 16.0),
        (7 * 16, 2.0 + 8.0 / 16.0),
        (8 * 16, 2.0 + 4.0 / 16.0),
    ];
    for (x, height) in expected {
        let y = surface[&(x, 4 * 16)];
        assert!(
            (y - height).abs() < 1e-6,
            "surface at x={x}/16 should be y={height}, got {y}",
        );
    }
}

#[test]
fn adjacent_cells_agree_on_every_shared_vertex() {
    // A source poured onto a floor and spread by Manhattan distance,
    // which is the diamond the flow rule actually produces.
    let mut blocks = floor();
    let (sx, sz) = (6i32, 6i32);
    for x in 0..12i32 {
        for z in 0..12i32 {
            let distance = (x - sx).abs() + (z - sz).abs();
            if distance <= MAX_STRENGTH as i32 {
                blocks.push((x as u32, 2, z as u32, FluidBlock::Water(distance as u8)));
            }
        }
    }

    // `surface_map` panics on any disagreement. Run it both ways: greedy
    // merging must not move a vertex either.
    let greedy = surface_map(&mesh_with(&blocks, true));
    let ungreedy = surface_map(&mesh_with(&blocks, false));
    assert_eq!(greedy.len(), ungreedy.len());
    for (corner, y) in &ungreedy {
        assert!(
            (greedy[corner] - y).abs() < 1e-6,
            "corner {corner:?} moved when greedy meshing was enabled",
        );
    }

    // The outer rim of the diamond is the shallowest strength, so the
    // lowest point of the surface is that strength's height.
    let lowest = ungreedy.values().copied().fold(f32::INFINITY, f32::min);
    assert!(
        (lowest - (2.0 + HEIGHTS[MAX_STRENGTH as usize] as f32 / 16.0)).abs() < 1e-6,
        "the edge of the diamond should be the lowest height, got {lowest}",
    );
}

#[test]
fn side_faces_meet_the_surface_they_border() {
    let mut blocks = floor();
    blocks.extend([
        (4, 2, 4, FluidBlock::Water(0)),
        (5, 2, 4, FluidBlock::Water(1)),
        (6, 2, 4, FluidBlock::Water(2)),
    ]);
    let scene = mesh_with(&blocks, false);
    let surface = surface_map(&scene);

    for face in [
        AlignedFace::PosX,
        AlignedFace::NegX,
        AlignedFace::PosZ,
        AlignedFace::NegZ,
    ] {
        for quad in scene.fluid_quads(face) {
            for (x, z, y) in corners(quad, face) {
                // The two vertices at the cell floor sit on the block
                // boundary; the two at the top must land exactly on the
                // surface the top faces drew.
                if (y - 2.0).abs() < 1e-6 {
                    continue;
                }
                let expected = surface[&(x, z)];
                assert!(
                    (y - expected).abs() < 1e-6,
                    "side face {face:?} at ({x}, {z}) reaches y={y}, \
                     but the surface there is y={expected}",
                );
            }
        }
    }
}

#[test]
fn water_under_water_fills_its_cell() {
    let mut blocks = floor();
    blocks.extend([
        // A shallow strength stacked two deep: the lower cell has
        // nothing above it to be a surface of, so it is full whatever
        // it claims.
        (4, 2, 4, FluidBlock::Water(3)),
        (4, 3, 4, FluidBlock::Water(3)),
    ]);
    let scene = mesh_with(&blocks, false);

    let lower_side = scene.quad_at(AlignedFace::PosX, UVec3::new(4, 2, 4));
    assert_eq!(
        lower_side.corner_offsets, [0; 4],
        "a submerged column should reach the top of its cell",
    );

    assert!(
        !scene
            .fluid_quads(AlignedFace::PosY)
            .any(|quad| quad.voxel_position(AlignedFace::PosY) == UVec3::new(4, 2, 4)),
        "the submerged cell has no surface to draw",
    );

    let upper_top = scene.quad_at(AlignedFace::PosY, UVec3::new(4, 3, 4));
    for (_, _, y) in corners(upper_top, AlignedFace::PosY) {
        assert!(
            (y - (3.0 + 4.0 / 16.0)).abs() < 1e-6,
            "the top cell should stand at its own height, got y={y}",
        );
    }
}

#[test]
fn shallow_surface_under_a_ceiling_is_not_culled() {
    let mut blocks = floor();
    blocks.extend([
        (4, 2, 4, FluidBlock::Water(3)),
        // A ceiling one cell up. The surface is 4/16ths off the floor,
        // so there is a gap to see it through.
        (4, 3, 4, FluidBlock::Stone),
    ]);
    let scene = mesh_with(&blocks, false);

    assert!(
        scene
            .fluid_quads(AlignedFace::PosY)
            .any(|quad| quad.voxel_position(AlignedFace::PosY) == UVec3::new(4, 2, 4)),
        "a shallow surface is visible under a ceiling and must be drawn",
    );
}

#[test]
fn lava_does_not_stitch_onto_water() {
    let mut blocks = floor();
    blocks.extend([(4, 2, 4, FluidBlock::Water(3)), (5, 2, 4, FluidBlock::Lava)]);
    let scene = mesh_with(&blocks, false);

    let water = scene.quad_at(AlignedFace::PosY, UVec3::new(4, 2, 4));
    for (x, z, y) in corners(water, AlignedFace::PosY) {
        assert!(
            (y - (2.0 + 4.0 / 16.0)).abs() < 1e-6,
            "water at ({x}, {z}) should keep its own height beside lava, got y={y}",
        );
    }
}

#[test]
fn standing_water_merges_greedily() {
    let mut blocks = floor();
    for x in 4..8 {
        for z in 4..8 {
            blocks.push((x, 2, z, FluidBlock::Water(0)));
        }
    }
    let scene = mesh_with(&blocks, true);

    let tops: Vec<_> = scene.fluid_quads(AlignedFace::PosY).collect();
    assert_eq!(
        tops.len(),
        1,
        "a flat 4x4 of sources should merge into one quad",
    );
    assert_eq!(tops[0].size, glam::UVec2::new(64, 64));
    assert_eq!(tops[0].corner_offsets, [0; 4]);
}

#[test]
fn a_slope_merges_along_its_level_direction_only() {
    let mut blocks = floor();
    // A wall of sources with a strip of shallower flow beside it: every
    // cell of the strip is level along z and sloped along x.
    for z in 4..8 {
        blocks.push((4, 2, z, FluidBlock::Water(0)));
        blocks.push((5, 2, z, FluidBlock::Water(1)));
    }
    let scene = mesh_with(&blocks, true);

    let strip = scene.quad_at(AlignedFace::PosY, UVec3::new(5, 2, 4));
    // u is Z for a Y face, so a strip merged along z is 64 wide and one
    // cell deep — it cannot merge into the sources, whose surface is
    // higher, nor stretch across its own slope.
    assert_eq!(strip.size, glam::UVec2::new(64, 16));

    let heights: Vec<f32> = corners(strip, AlignedFace::PosY)
        .into_iter()
        .map(|(_, _, y)| y)
        .collect();
    assert!(
        heights.iter().any(|&y| (y - 3.0).abs() < 1e-6)
            && heights
                .iter()
                .any(|&y| (y - (2.0 + 12.0 / 16.0)).abs() < 1e-6),
        "the merged strip should still slope from the sources down, got {heights:?}",
    );
}

#[test]
fn a_slope_does_not_merge_across_its_fall() {
    let mut blocks = floor();
    // A source with four orthogonal neighbours. Each neighbour slopes
    // away from the centre in a different direction, so no two of them
    // share an entry and none can merge.
    blocks.extend([
        (4, 2, 4, FluidBlock::Water(0)),
        (3, 2, 4, FluidBlock::Water(1)),
        (5, 2, 4, FluidBlock::Water(1)),
        (4, 2, 3, FluidBlock::Water(1)),
        (4, 2, 5, FluidBlock::Water(1)),
    ]);
    let scene = mesh_with(&blocks, true);

    assert_eq!(
        scene.fluid_quads(AlignedFace::PosY).count(),
        5,
        "each arm of the plus slopes its own way and must stay its own quad",
    );
}

#[test]
fn the_padding_ring_feeds_the_height_field() {
    // The same cell meshed twice: once with its source neighbour inside
    // the chunk, once with the neighbour only in the padding ring. A
    // chunk seam must not change the surface.
    let mut inside = floor();
    inside.extend([
        (4, 2, 4, FluidBlock::Water(0)),
        (5, 2, 4, FluidBlock::Water(1)),
    ]);
    let from_inside = mesh_with(&inside, false);
    let reference = from_inside
        .quad_at(AlignedFace::PosY, UVec3::new(5, 2, 4))
        .corner_offsets;

    let mut chunk = PaddedChunk16::new_filled(FluidBlock::Air);
    for z in 0..3 {
        chunk.set_padded(UVec3::new(0, 2, z), FluidBlock::Stone);
        chunk.set_padded(UVec3::new(1, 2, z), FluidBlock::Stone);
    }
    // Inner (0, 2, 0), with the source one step outside the chunk.
    chunk.set(UVec3::new(0, 2, 0), FluidBlock::Water(1));
    chunk.set_padded(UVec3::new(0, 3, 1), FluidBlock::Water(0));
    let from_padding = mesh_chunk(&chunk, false);

    let edge = from_padding.faces[AlignedFace::PosY.index()]
        .iter()
        .find(|quad| quad.voxel_position(AlignedFace::PosY) == UVec3::new(0, 2, 0))
        .expect("the edge cell has a surface");
    assert_eq!(
        edge.corner_offsets, reference,
        "a source in the padding ring should raise the surface the same way",
    );
}

#[test]
fn fluid_at_every_chunk_extreme_stays_in_bounds() {
    // The height field reads a 3x3 of columns and one step above each,
    // so a cell in the corner of the chunk reaches diagonally into the
    // padding ring. A checkerboard reaches every extreme of that ring
    // while leaving each cell's faces exposed, so the gather actually
    // runs rather than being culled away first — and the debug
    // assertions in the mesher speak for whether it stayed in bounds.
    let padded = 16u32 + 2;
    let mut chunk = PaddedChunk16::new_filled(FluidBlock::Air);
    for x in 0..padded {
        for y in 0..padded {
            for z in 0..padded {
                if (x + y + z) % 2 != 0 {
                    continue;
                }
                let strength = ((x + y + z) / 2 % (MAX_STRENGTH as u32 + 1)) as u8;
                chunk.set_padded(UVec3::new(x, y, z), FluidBlock::Water(strength));
            }
        }
    }

    for greedy in [false, true] {
        let quads = mesh_chunk(&chunk, greedy);
        assert!(
            quads.total() > 0,
            "isolated fluid cells all show their faces (greedy={greedy})",
        );
    }
}

#[test]
fn a_held_fluid_is_a_cube() {
    let quads = mesh_block(&FluidBlock::Water(3), ());

    assert_eq!(quads.total(), 6);
    for face in AlignedFace::ALL {
        assert_eq!(
            quads.faces[face.index()][0].corner_offsets,
            [0; 4],
            "face {face:?} of a held fluid should be a plain cube face",
        );
    }
}

#[test]
fn side_face_texture_is_cropped_not_squashed() {
    let mut blocks = floor();
    blocks.push((4, 2, 4, FluidBlock::Water(2)));
    let scene = mesh_with(&blocks, false);

    let side = scene.quad_at(AlignedFace::PosX, UVec3::new(4, 2, 4));
    let shape = FluidBlock::Water(2).shape();
    let uvs = side.texture_coordinates(AlignedFace::PosX, shape, Axis::X, false);
    let vs: Vec<f32> = uvs.iter().map(|uv| uv.y).collect();

    // The quad is half a block tall, so it samples half the texture
    // rather than the whole of it compressed.
    let (min, max) = vs
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, &v| {
            (acc.0.min(v), acc.1.max(v))
        });
    assert!(
        (min - 0.0).abs() < 1e-6,
        "expected v to start at 0, got {min}"
    );
    assert!(
        (max - 8.0 / 16.0).abs() < 1e-6,
        "expected v to stop at the surface (0.5), got {max}",
    );
}

#[test]
fn a_fluid_free_chunk_is_untouched() {
    // The whole point of `FLUID_ENABLED`: a consumer without fluids sees
    // exactly the geometry it saw before, with no offsets to ignore.
    let scene = mesh_with(&floor(), true);
    for face in AlignedFace::ALL {
        for quad in &scene.quads.faces[face.index()] {
            assert_eq!(quad.corner_offsets, [0; 4]);
        }
    }
}
