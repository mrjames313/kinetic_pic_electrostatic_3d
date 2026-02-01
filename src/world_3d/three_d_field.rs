use anyhow::Result;
use glam::DVec3;
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign, Mul};

// Consider using the ndarray crate instead of custom implementation below

pub struct ThreeDField <T> {
    nx: usize,
    ny: usize,
    nz: usize,
    data: Vec<T>,
}

impl<T> ThreeDField <T> 
    where
    T: Clone + Copy + AddAssign + SubAssign + MulAssign + DivAssign +
    MulAssign<f64> + Mul<f64, Output = T>, f64 : Mul<T>
{

    pub fn init(nx: usize, ny: usize, nz: usize, val: T) -> Self {
        let f = ThreeDField {nx, ny, nz, data: vec![val; nx * ny * nz],};
        f
    }

    pub fn set_all(&mut self, val : T) {
        self.data.fill(val);
    }
    
    fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        iz * self.nx * self.ny + iy * self.nx + ix
    }

    pub fn get(&self, ix: usize, iy: usize, iz: usize) ->  T {
        self.data[self.idx(ix, iy, iz)]
    }

    pub fn set(&mut self, ix: usize, iy: usize, iz: usize, val: T) {
        let idx = self.idx(ix, iy, iz);
        self.data[idx] = val;
    }

    // Like set, but distributes to neighboring nodes according to
    // volumetric proportion
    // full_idx has both the integer and fractional components
    // TODO: add testing
    pub fn distribute(&mut self, full_idx : DVec3, value : T) {
        // TODO: add asserts on bounds, will introduce a return for error
        let ix = full_idx[0] as usize;
        let fix = full_idx[0] - (ix as f64);
        let iy = full_idx[1] as usize;
        let fiy = full_idx[1] - (iy as f64);
        let iz = full_idx[2] as usize;
        let fiz = full_idx[2] - (iz as f64);

        self.set(ix,   iy,   iz,   value * (1.0 - fix) * (1.0 - fiy) * (1.0 - fiz));
        self.set(ix,   iy,   iz+1, value * (1.0 - fix) * (1.0 - fiy) * (fiz));
        self.set(ix,   iy+1, iz,   value * (1.0 - fix) * (fiy) *       (1.0 - fiz));
        self.set(ix,   iy+1, iz+1, value * (1.0 - fix) * (fiy) *       (fiz));
        self.set(ix+1, iy,   iz,   value * (fix) *       (1.0 - fiy) * (1.0 - fiz));
        self.set(ix+1, iy,   iz+1, value * (fix) *       (1.0 - fiy) * (fiz));
        self.set(ix+1, iy+1, iz,   value * (fix) *       (fiy) *       (1.0 - fiz));
        self.set(ix+1, iy+1, iz+1, value * (fix) *       (fiy) *       (fiz));
    }
    
    pub fn len(&self) -> usize { self.data.len() }

    pub fn data(&self) -> &[T] { &self.data }
    
    pub fn elementwise_inplace_add(&mut self, other: &ThreeDField<T>) {
        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x += *y;
        }
    }

    pub fn elementwise_inplace_sub(&mut self, other: &ThreeDField<T>) {
        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x -= *y;
        }
    }

    pub fn elementwise_inplace_mult(&mut self, other: &ThreeDField<T>) {
        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x *= *y;
        }
    }

    pub fn elementwise_inplace_add_scaled(&mut self, scale : f64,
                                              other: &ThreeDField<T>) {
        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x += *y * scale;
        }
    }

    pub fn elementwise_inplace_div(&mut self, other: &ThreeDField<T>) {
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


