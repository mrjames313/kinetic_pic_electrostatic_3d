use glam::DVec3;
use std::ops::{Add, AddAssign, SubAssign, MulAssign, DivAssign, Mul};


pub struct ThreeDField <T> {
    nx: usize,
    ny: usize,
    nz: usize,
    data: Vec<T>,
}

impl<T> ThreeDField <T> 
    where
    T: Clone + Copy + Add<Output = T> + AddAssign + SubAssign + MulAssign + DivAssign +
    MulAssign<f64> + Mul<f64, Output = T>, f64 : Mul<T>
{

    pub fn new(nx: usize, ny: usize, nz: usize, val: T) -> anyhow::Result<Self> {
        Ok( Self {nx, ny, nz, data: vec![val; nx * ny * nz], } )
    }

    pub fn set_all(&mut self, val : T) {
        self.data.fill(val);
    }

    // TODO: consider the overhead of these checks
    // TODO: create testing around this?
    fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        assert!(ix < self.nx, "x index ({ix}) out of bounds [0, {})", self.nx);
        assert!(iy < self.ny, "y index ({iy}) out of bounds [0, {})", self.ny);
        assert!(iz < self.nz, "z index ({iz}) out of bounds [0, {})", self.nz);
        
        iz * self.nx * self.ny + iy * self.nx + ix
    }

    pub fn get(&self, ix: usize, iy: usize, iz: usize) ->  T {
        self.data[self.idx(ix, iy, iz)]
    }

    pub fn set(&mut self, ix: usize, iy: usize, iz: usize, val: T) {
        let idx = self.idx(ix, iy, iz);
        self.data[idx] = val;
    }

    pub fn add(&mut self, ix: usize, iy: usize, iz: usize, val: T) {
        let idx = self.idx(ix, iy, iz);
        self.data[idx] += val;
    }

    // Like set, but distributes to neighboring nodes according to
    // volumetric proportion
    // full_idx has both the integer and fractional components, and
    // it must be strictly less than the upper index boundary
    pub fn distribute(&mut self, full_idx : DVec3, value : T) {
        assert!(full_idx[0] >= 0.0 && full_idx[0] < (self.nx-1) as f64,
                "x index {} out of bounds [0, {})",
                full_idx[0], self.nx-1);
        assert!(full_idx[1] >= 0.0 && full_idx[1] < (self.ny-1) as f64,
                "y index {} out of bounds [0, {})",
                full_idx[1], self.ny-1);
        assert!(full_idx[2] >= 0.0 && full_idx[2] < (self.nz-1) as f64,
                "z index {} out of bounds [0, {})",
                full_idx[2], self.nz-1);
        
        // TODO: add asserts on bounds, will introduce a return for error
        let ix = full_idx[0] as usize;
        let fix = full_idx[0] - (ix as f64);
        let iy = full_idx[1] as usize;
        let fiy = full_idx[1] - (iy as f64);
        let iz = full_idx[2] as usize;
        let fiz = full_idx[2] - (iz as f64);

        self.add(ix,   iy,   iz,   value * (1.0 - fix) * (1.0 - fiy) * (1.0 - fiz));
        self.add(ix,   iy,   iz+1, value * (1.0 - fix) * (1.0 - fiy) * (fiz));
        self.add(ix,   iy+1, iz,   value * (1.0 - fix) * (fiy) *       (1.0 - fiz));
        self.add(ix,   iy+1, iz+1, value * (1.0 - fix) * (fiy) *       (fiz));
        self.add(ix+1, iy,   iz,   value * (fix) *       (1.0 - fiy) * (1.0 - fiz));
        self.add(ix+1, iy,   iz+1, value * (fix) *       (1.0 - fiy) * (fiz));
        self.add(ix+1, iy+1, iz,   value * (fix) *       (fiy) *       (1.0 - fiz));
        self.add(ix+1, iy+1, iz+1, value * (fix) *       (fiy) *       (fiz));
    }

    pub fn linear_interpolate(&self, full_idx : DVec3) -> T {
        assert!(full_idx[0] >= 0.0 && full_idx[0] < (self.nx-1) as f64,
                "x index {} out of bounds [0, {})",
                full_idx[0], self.nx-1);
        assert!(full_idx[1] >= 0.0 && full_idx[1] < (self.ny-1) as f64,
                "y index {} out of bounds [0, {})",
                full_idx[1], self.ny-1);
        assert!(full_idx[2] >= 0.0 && full_idx[2] < (self.nz-1) as f64,
                "z index {} out of bounds [0, {})",
                full_idx[2], self.nz-1);

        let ix = full_idx[0] as usize;
        let fix = full_idx[0] - (ix as f64);
        let iy = full_idx[1] as usize;
        let fiy = full_idx[1] - (iy as f64);
        let iz = full_idx[2] as usize;
        let fiz = full_idx[2] - (iz as f64);

        let val : T = self.get(ix, iy, iz) * (1.0 - fix) * (1.0 - fiy) * (1.0 - fiz) +
            self.get(ix, iy, iz+1)         * (1.0 - fix) * (1.0 - fiy) *  fiz +
            self.get(ix, iy+1, iz)         * (1.0 - fix) * fiy         * (1.0 - fiz) +
            self.get(ix, iy+1, iz+1)       * (1.0 - fix) * fiy         * fiz +
            self.get(ix+1, iy, iz)         * fix         * (1.0 - fiy) * (1.0 - fiz) +
            self.get(ix+1, iy, iz+1)       * fix         * (1.0 - fiy) * fiz +
            self.get(ix+1, iy+1, iz)       * fix         * fiy         * (1.0 - fiz) +
            self.get(ix+1, iy+1, iz+1)     * fix         * fiy         * fiz ;
        val
    }
    
    
    pub fn len(&self) -> usize { self.data.len() }

    pub fn data(&self) -> &[T] { &self.data }
    
    pub fn elementwise_inplace_add(&mut self, other: &ThreeDField<T>) {
        assert_eq!(self.data.len(), other.data.len(), "length mismatch");
        assert!(self.nx == other.nx && self.ny == other.ny && self.nz == other.nz,
                "dimension mismatch");

        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x += *y;
        }
    }

    pub fn elementwise_inplace_sub(&mut self, other: &ThreeDField<T>) {
        assert_eq!(self.data.len(), other.data.len(), "dimension mismatch");
        assert!(self.nx == other.nx && self.ny == other.ny && self.nz == other.nz,
                "dimension mismatch");

        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x -= *y;
        }
    }

    pub fn elementwise_inplace_mult(&mut self, other: &ThreeDField<T>) {
        assert_eq!(self.data.len(), other.data.len(), "dimension mismatch");
        assert!(self.nx == other.nx && self.ny == other.ny && self.nz == other.nz,
                "dimension mismatch");

        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x *= *y;
        }
    }

    pub fn elementwise_inplace_add_scaled(&mut self, scale : f64,
                                          other: &ThreeDField<T>) {
        assert_eq!(self.data.len(), other.data.len(), "dimension mismatch");
        assert!(self.nx == other.nx && self.ny == other.ny && self.nz == other.nz,
                "dimension mismatch");

        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x += *y * scale;
        }
    }

    pub fn elementwise_inplace_div(&mut self, other: &ThreeDField<T>) {
        assert_eq!(self.data.len(), other.data.len(), "dimension mismatch");
        assert!(self.nx == other.nx && self.ny == other.ny && self.nz == other.nz,
                "dimension mismatch");
        
        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x /= *y;
        }
    }

//    fn scalar_inplace_add(&mut self, s: f64) {
//        for x in self.data.iter_mut() {
//            *x += s;
//        }
//    }

    pub fn scalar_inplace_mult(&mut self, s: f64) {
        for x in self.data.iter_mut() {
            *x *= s;
        }
    }
    
}


#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn set_get_roundtrip() {
        let mut f = ThreeDField::<f64>::new(4, 3, 2, 0.0).unwrap();
        f.set(0, 0, 0, 1.0);
        f.set(3, 2, 1, 7.5);
        f.set(1, 2, 0, -2.0);

        assert_eq!(f.get(0, 0, 0), 1.0);
        assert_eq!(f.get(3, 2, 1), 7.5);
        assert_eq!(f.get(1, 2, 0), -2.0);
    }

    #[test]
    #[should_panic]
    fn get_panics_on_oob_x() {
        let f = ThreeDField::<f64>::new(4, 3, 2, 0.0).unwrap();
        let _ = f.get(4, 0, 0);
    }

    #[test]
    #[should_panic]
    fn get_panics_on_oob_y() {
        let f = ThreeDField::<f64>::new(4, 3, 2, 0.0).unwrap();
        let _ = f.get(0, 3, 0);
    }

    #[test]
    #[should_panic]
    fn get_panics_on_oob_z() {
        let f = ThreeDField::<f64>::new(4, 3, 2, 0.0).unwrap();
        let _ = f.get(0, 0, 2);
    }

    #[test]
    fn set_all_fills_everywhere() {
        let mut f = ThreeDField::<f64>::new(3, 2, 4, 0.0).unwrap();
        f.set_all(2.5);
        assert_eq!(f.len(), 3 * 2 * 4);
        assert!(f.data().iter().all(|&x| x == 2.5));
    }

    #[test]
    fn add_accumulates() {
        let mut f = ThreeDField::<f64>::new(2, 2, 2, 0.0).unwrap();
        f.add(1, 0, 1, 1.5);
        f.add(1, 0, 1, 2.0);
        assert_eq!(f.get(1, 0, 1), 3.5);
    }

    fn sum_field(f: &ThreeDField<f64>) -> f64 {
        f.data().iter().sum()
    }

    #[test]
    fn distribute_conserves_sum_interior() {
        let mut f = ThreeDField::<f64>::new(4, 4, 4, 0.0).unwrap();
        let value = 10.0;
        let p = DVec3::new(1.25, 2.5, 1.75); // interior: ix=1,iy=2,iz=1; +1 still in-bounds

        f.distribute(p, value);  // Doesn't make any assumptions about where to distribute, just conservation

        let total = sum_field(&f);
        assert!((total - value).abs() < 1e-12);
    }

    #[test]
    fn distribute_on_node_goes_to_single_cell() {
        let mut f = ThreeDField::<f64>::new(4, 4, 4, 0.0).unwrap();
        let value = 3.0;
        let p = DVec3::new(1.0, 2.0, 1.0);

        f.distribute(p, value);

        assert_eq!(f.get(1, 2, 1), value);
        // Everything else stays zero
        let total = sum_field(&f);
        assert!((total - value).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn distribute_panics_near_upper_boundary() {
        let mut f = ThreeDField::<f64>::new(4, 4, 4, 0.0).unwrap();
        // ix=3 => ix+1 out of bounds
        let _ = f.distribute(DVec3::new(3.1, 1.2, 1.2), 6.5);
    }

    #[test]
    #[should_panic]
    fn distribute_panics_beyond_lower_boundary() {
        let mut f = ThreeDField::<f64>::new(4, 4, 4, 0.0).unwrap();
        // ix=2 < 0 out of bounds
        let _ = f.distribute(DVec3::new(3.1, -0.4, 1.2), -2.5);
    }

    #[test]
    fn linear_interpolate_constant_field() {
        let mut f = ThreeDField::<f64>::new(4, 4, 4, 0.0).unwrap();
        f.set_all(7.0);

        let v = f.linear_interpolate(DVec3::new(1.2, 2.7, 0.3));
        assert!((v - 7.0).abs() < 1e-12);
    }

    #[test]
    fn linear_interpolate_exact_on_nodes() {
        let mut f = ThreeDField::<f64>::new(4, 4, 4, 0.0).unwrap();
        f.set(2, 1, 2, 9.5);

        let v = f.linear_interpolate(DVec3::new(2.0, 1.0, 2.0));
        assert!((v - 9.5).abs() < 1e-12);
    }

    #[test]
    fn linear_interpolate_reproduces_linear_field() {
        let (nx, ny, nz) = (5, 6, 7);
        let mut f = ThreeDField::<f64>::new(nx, ny, nz, 0.0).unwrap();

        let ax = 1.2;
        let ay = -0.7;
        let az = 0.3;
        let c = 2.0;

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let val = ax * (ix as f64) + ay * (iy as f64) + az * (iz as f64) + c;
                    f.set(ix, iy, iz, val);
                }
            }
        }

        // Choose an interior point so ix+1,iy+1,iz+1 are valid
        let p = DVec3::new(2.25, 3.5, 4.75);
        let expected = ax * p.x + ay * p.y + az * p.z + c;

        let got = f.linear_interpolate(p);
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn linear_interpolate_panics_near_upper_boundary() {
        let f = ThreeDField::<f64>::new(4, 4, 4, 0.0).unwrap();
        // ix=3 => ix+1 out of bounds
        let _ = f.linear_interpolate(DVec3::new(3.1, 1.2, 1.2));
    }

    #[test]
    #[should_panic]
    fn linear_interpolate_panics_beyond_lower_boundary() {
        let f = ThreeDField::<f64>::new(4, 4, 4, 0.0).unwrap();
        // ix=1 < 0 out of bounds
        let _ = f.linear_interpolate(DVec3::new(-3.1, 1.2, 1.2));
    }

    #[test]
    fn elementwise_inplace_add_works() {
        let mut a = ThreeDField::<f64>::new(2, 2, 2, 1.0).unwrap();
        let b = ThreeDField::<f64>::new(2, 2, 2, 2.0).unwrap();

        a.elementwise_inplace_add(&b);
        assert!(a.data().iter().all(|&x| x == 3.0));
    }

    #[test]
    #[should_panic]
    fn elementwise_add_panics_on_mismatch() {
        let mut a = ThreeDField::<f64>::new(2, 2, 2, 1.0).unwrap();
        let b = ThreeDField::<f64>::new(3, 2, 2, 2.0).unwrap();
        a.elementwise_inplace_add(&b);
    }

    #[test]
    fn elementwise_inplace_sub_works() {
        let mut a = ThreeDField::<f64>::new(2, 2, 2, 5.0).unwrap();
        let b = ThreeDField::<f64>::new(2, 2, 2, 2.0).unwrap();

        a.elementwise_inplace_sub(&b);
        assert!(a.data().iter().all(|&x| x == 3.0));
    }

    #[test]
    fn elementwise_inplace_mult_works() {
        let mut a = ThreeDField::<f64>::new(2, 2, 2, 3.0).unwrap();
        let b = ThreeDField::<f64>::new(2, 2, 2, 2.0).unwrap();

        a.elementwise_inplace_mult(&b);
        assert!(a.data().iter().all(|&x| x == 6.0));
    }

    #[test]
    fn elementwise_inplace_div_works() {
        let mut a = ThreeDField::<f64>::new(2, 2, 2, 6.0).unwrap();
        let b = ThreeDField::<f64>::new(2, 2, 2, 2.0).unwrap();

        a.elementwise_inplace_div(&b);
        assert!(a.data().iter().all(|&x| x == 3.0));
    }

    #[test]
    fn elementwise_inplace_add_scaled_works() {
        let mut a = ThreeDField::<f64>::new(2, 2, 2, 1.0).unwrap();
        let b = ThreeDField::<f64>::new(2, 2, 2, 3.0).unwrap();

        a.elementwise_inplace_add_scaled(2.0, &b); // a = 1 + 2*3 = 7
        assert!(a.data().iter().all(|&x| x == 7.0));
    }

    #[test]
    fn scalar_inplace_mult_works() {
        let mut a = ThreeDField::<f64>::new(2, 2, 2, 1.5).unwrap();
        a.scalar_inplace_mult(3.0);
        assert!(a.data().iter().all(|&x| x == 4.5));
    }

    #[test]
    #[should_panic]
    fn elementwise_sub_panics_on_mismatch() {
        let mut a = ThreeDField::<f64>::new(2, 2, 2, 1.0).unwrap();
        let b = ThreeDField::<f64>::new(2, 3, 2, 1.0).unwrap();
        a.elementwise_inplace_sub(&b);
    }

    #[test]
    #[should_panic]
    fn elementwise_mult_panics_on_mismatch() {
        let mut a = ThreeDField::<f64>::new(2, 2, 2, 1.0).unwrap();
        let b = ThreeDField::<f64>::new(2, 2, 3, 1.0).unwrap();
        a.elementwise_inplace_mult(&b);
    }

    #[test]
    #[should_panic]
    fn elementwise_div_panics_on_mismatch() {
        let mut a = ThreeDField::<f64>::new(2, 2, 2, 1.0).unwrap();
        let b = ThreeDField::<f64>::new(1, 2, 2, 1.0).unwrap();
        a.elementwise_inplace_div(&b);
    }

    #[test]
    #[should_panic]
    fn elementwise_add_scaled_panics_on_mismatch() {
        let mut a = ThreeDField::<f64>::new(2, 2, 2, 1.0).unwrap();
        let b = ThreeDField::<f64>::new(2, 2, 1, 1.0).unwrap();
        a.elementwise_inplace_add_scaled(1.0, &b);
    }

}
