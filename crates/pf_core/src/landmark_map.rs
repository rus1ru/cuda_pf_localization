//! Landmark map: dense id-indexed 3D positions.

use nalgebra::Vector3;

/// id -> 3D position. Ids are dense non-negative integers.
#[derive(Debug, Clone, Default)]
pub struct LandmarkMap {
    positions: Vec<Option<Vector3<f64>>>,
}

impl LandmarkMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a landmark.
    pub fn add(&mut self, id: i32, position: Vector3<f64>) {
        assert!(id >= 0, "landmark ids must be non-negative");
        let id = id as usize;
        if self.positions.len() <= id {
            self.positions.resize(id + 1, None);
        }
        self.positions[id] = Some(position);
    }

    pub fn has(&self, id: i32) -> bool {
        id >= 0 && (id as usize) < self.positions.len() && self.positions[id as usize].is_some()
    }

    /// Panics if the id is absent; check [`has`](Self::has) first.
    pub fn at(&self, id: i32) -> Vector3<f64> {
        self.positions[id as usize].expect("landmark id not present")
    }

    /// Highest id + 1 (buffer size the CUDA backend needs).
    pub fn capacity(&self) -> usize {
        self.positions.len()
    }

    /// Number of defined landmarks.
    pub fn len_defined(&self) -> usize {
        self.positions.iter().filter(|p| p.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len_defined() == 0
    }

    /// Iterate (id, position) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (i32, Vector3<f64>)> + '_ {
        self.positions
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|v| (i as i32, v)))
    }

    /// Load from a YAML-ish text: lines of `id: [x, y, z]` (also accepts
    /// `id x y z`). `#` starts a comment.
    pub fn load_from_str(&mut self, text: &str) -> Result<usize, String> {
        let mut count = 0;
        for (lineno, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let cleaned = line.replace(['[', ']', ':', ','], " ");
            let nums: Vec<&str> = cleaned.split_whitespace().collect();
            if nums.len() != 4 {
                return Err(format!("line {}: expected 'id: [x, y, z]'", lineno + 1));
            }
            let parse = |s: &str| s.parse::<f64>().map_err(|e| format!("line {}: {}", lineno + 1, e));
            let id: i32 = parse(nums[0])? as i32;
            let x = parse(nums[1])?;
            let y = parse(nums[2])?;
            let z = parse(nums[3])?;
            self.add(id, Vector3::new(x, y, z));
            count += 1;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn add_has_at() {
        let mut m = LandmarkMap::new();
        assert!(!m.has(3));
        m.add(3, Vector3::new(1.0, 2.0, 3.0));
        assert!(m.has(3));
        assert_relative_eq!(m.at(3).x, 1.0);
        assert_eq!(m.len_defined(), 1);
    }

    #[test]
    fn parse_map_text() {
        let mut m = LandmarkMap::new();
        let n = m
            .load_from_str("# comment\n0: [1, 2, 3]\n5: [4.0, 5.0, 6.0]\n")
            .unwrap();
        assert_eq!(n, 2);
        assert!(m.has(0) && m.has(5));
        assert_relative_eq!(m.at(5).z, 6.0);
    }

    #[test]
    fn parse_rejects_garbage() {
        let mut m = LandmarkMap::new();
        assert!(m.load_from_str("0 [1, 2]\n").is_err());
    }
}
