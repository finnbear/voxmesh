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
}

impl Light for () {
    type Average = ();

    #[inline]
    fn average(_values: &[()]) {}
}

impl Light for u8 {
    type Average = f32;

    #[inline]
    fn average(values: &[u8]) -> f32 {
        let sum: u16 = values.iter().map(|&v| v as u16).sum();
        sum as f32 / values.len() as f32
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
}
