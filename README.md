# voxmesh

Voxel greedy meshing library

## Features
- Block types
  - Whole blocks (e.g. grass)
  - Arbitrary thickness/direction slabs
  - Arbitrarily inset blocks (e.g. cactus, bamboo)
  - Arbitrary direction cross-shaped billboard blocks (e.g. shrub)
  - Arbitrary direction facade-shaped billboard blocks (e.g. rail, ladder)
- Vertex computations
  - Positions
  - Normals
  - Texture coordinates
  - Smooth lighting
  - Ambient occlusion
  - Indices
- Mesh a single block or a chunk

## Assumptions

- 16x16x16 block chunks with 1 block of padding on each side
- OpenGL coordinate system
- Blocks are 1 unit wide
- Sub-block positions and sizes are in units of 1/16
- Merging is based on simple equality comparison
- You must propagate per-voxel light in advance
