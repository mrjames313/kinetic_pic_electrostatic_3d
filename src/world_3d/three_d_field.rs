use anyhow::Result;
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign};

// Consider using the ndarray crate instead of custom implementation below

pub struct ThreeDField <T> {
    nx: usize,
    ny: usize,
    nz: usize,
    data: Vec<T>,
}

impl<T> ThreeDField <T> 
    where
    T: Clone + Copy + AddAssign + SubAssign + MulAssign + DivAssign + MulAssign<f64>
{

    pub fn init(nx: usize, ny: usize, nz: usize, val: T) -> Self {
        let f = ThreeDField {nx, ny, nz, data: vec![val; nx * ny * nz],};
        f
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


