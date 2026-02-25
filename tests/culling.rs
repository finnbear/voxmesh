mod common;

use common::*;

#[test]
fn two_adjacent_opaque_blocks_cull_shared_face() {
    let q = mesh_with(&[(0, 0, 0, TestBlock::Stone), (1, 0, 0, TestBlock::Stone)]);
    // Shared PosX/NegX culled. Remaining coplanar faces merge.
    // PosX:1 + NegX:1 + PosY:1 + NegY:1 + PosZ:1 + NegZ:1 = 6
    assert_eq!(q.total(), 6);
}

#[test]
fn two_different_opaque_blocks_still_cull_shared_face() {
    let q = mesh_with(&[(0, 0, 0, TestBlock::Stone), (1, 0, 0, TestBlock::Dirt)]);
    // Shared face culled (opaque-opaque), but coplanar faces can't merge
    // across different block types: 2 quads each on 4 side faces + 2 end caps.
    assert_eq!(q.total(), 10);
}

#[test]
fn glass_next_to_identical_glass_culls_shared_face() {
    let q = mesh_with(&[(0, 0, 0, TestBlock::Glass), (1, 0, 0, TestBlock::Glass)]);
    // TransparentMerged between identical blocks, shared face culled.
    assert_eq!(q.total(), 6);
}

#[test]
fn glass_next_to_stone_does_not_cull_stone() {
    let q = mesh_with(&[(0, 0, 0, TestBlock::Glass), (1, 0, 0, TestBlock::Stone)]);
    // Stone (Opaque) culls glass's PosX. Glass (TransparentMerged) does not
    // cull stone's NegX (different blocks). 5 + 6 = 11 quads (no merging).
    assert_eq!(q.total(), 11);
}

#[test]
fn leaves_never_cull_neighbor() {
    let q = mesh_with(&[(0, 0, 0, TestBlock::Leaves), (1, 0, 0, TestBlock::Leaves)]);
    // TransparentUnmerged never culls, both blocks keep all 6 faces.
    // Coplanar same-type faces merge on PosY/NegY/PosZ/NegZ (4 merged).
    // PosX:1 + NegX:1 + internal PosX:1 + internal NegX:1 + 4 merged = 8.
    assert_eq!(q.total(), 8);
}
