use std::fmt::Debug;
use std::mem;

/// Per-vertex light value that can be averaged across neighboring voxels.
///
/// Implement this trait for your light type to enable smooth per-vertex
/// lighting and ambient occlusion during greedy meshing.
///
/// The `()` implementation disables all lighting computation at zero cost.
pub trait Light: Copy + PartialEq + Eq + Debug + Default {
    /// Whether ambient occlusion computation is enabled.
    /// When false, all AO values are set to 3 (fully lit).
    const AO_ENABLED: bool = true;

    /// Whether smooth light computation is enabled.
    /// When false, all light values are set to the default.
    /// Defaults to `true` for non-zero-sized types.
    const LIGHT_ENABLED: bool = mem::size_of::<Self>() > 0;

    /// The type returned by [`average`](Self::average).
    ///
    /// For integer light types like `u8` this is typically `f32` to
    /// preserve fractional precision; for `()` it stays `()`.
    type Average: Copy + Default + Debug + PartialEq;

    /// Average the given light values.
    fn average(values: &[Self]) -> Self::Average;

    /// Blend four corner averages by weights that sum to one.
    ///
    /// The smooth-light field is defined at a face's four corners, and
    /// geometry that does not fill its cell has vertices part-way across
    /// it — a stair's tread, the strip a slab shows on its side. Those
    /// vertices take the field's value where they actually are, which is
    /// the bilinear blend of the four; handing them a corner's value
    /// instead stretches a whole cell's gradient over half a cell.
    ///
    /// `corners` is in the mask-local order the mesher works in,
    /// `[umin/vmin, umax/vmin, umax/vmax, umin/vmax]`, and `weights`
    /// matches it.
    fn blend(corners: &[Self::Average; 4], weights: [f32; 4]) -> Self::Average;
}

impl Light for () {
    type Average = ();

    #[inline]
    fn average(_values: &[()]) {}

    #[inline]
    fn blend(_corners: &[(); 4], _weights: [f32; 4]) {}
}

impl Light for u8 {
    type Average = f32;

    #[inline]
    fn average(values: &[u8]) -> f32 {
        let sum: u16 = values.iter().map(|&v| v as u16).sum();
        sum as f32 / values.len() as f32
    }

    #[inline]
    fn blend(corners: &[f32; 4], weights: [f32; 4]) -> f32 {
        corners[0] * weights[0]
            + corners[1] * weights[1]
            + corners[2] * weights[2]
            + corners[3] * weights[3]
    }
}

impl Light for [u8; 2] {
    type Average = [f32; 2];

    #[inline]
    fn average(values: &[[u8; 2]]) -> [f32; 2] {
        let len = values.len() as f32;
        let mut sums = [0u16; 2];
        for v in values {
            sums[0] += v[0] as u16;
            sums[1] += v[1] as u16;
        }
        [sums[0] as f32 / len, sums[1] as f32 / len]
    }

    #[inline]
    fn blend(corners: &[[f32; 2]; 4], weights: [f32; 4]) -> [f32; 2] {
        let mut out = [0.0; 2];
        for (corner, weight) in corners.iter().zip(weights) {
            out[0] += corner[0] * weight;
            out[1] += corner[1] * weight;
        }
        out
    }
}
