use std::fmt::Debug;

/// Per-vertex light value that can be averaged across neighboring voxels.
///
/// Implement this trait for your light type to enable smooth per-vertex
/// lighting and ambient occlusion during greedy meshing.
///
/// The `()` implementation disables all lighting computation at zero cost.
pub trait Light: Copy + PartialEq + Eq + Debug + Default {
    /// Whether AO and light computation is enabled.
    /// When false, the compiler eliminates all lighting code paths.
    const ENABLED: bool;

    /// Average up to 4 light values. `values[0..count]` are the valid entries.
    fn average(values: [Self; 4], count: u8) -> Self;
}

impl Light for () {
    const ENABLED: bool = false;

    #[inline]
    fn average(_values: [(); 4], _count: u8) {}
}

impl Light for u8 {
    const ENABLED: bool = true;

    #[inline]
    fn average(values: [u8; 4], count: u8) -> u8 {
        let sum: u16 = values[..count as usize].iter().map(|&v| v as u16).sum();
        (sum / count as u16) as u8
    }
}

impl Light for [u8; 2] {
    const ENABLED: bool = true;

    #[inline]
    fn average(values: [[u8; 2]; 4], count: u8) -> [u8; 2] {
        let c = count as u16;
        let mut sums = [0u16; 2];
        for v in &values[..count as usize] {
            sums[0] += v[0] as u16;
            sums[1] += v[1] as u16;
        }
        [(sums[0] / c) as u8, (sums[1] / c) as u8]
    }
}
