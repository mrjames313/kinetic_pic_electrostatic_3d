use anyhow::Result;


// Consider using the ndarray crate instead of custom implementation below

pub struct ThreeDField {
    nx: usize,
    ny: usize,
    nz: usize,
    data: Vec<f64>,
}

impl ThreeDField {

    pub fn init(nx: usize, ny: usize, nz: usize, val: f64) -> Self {
        let f = ThreeDField {nx, ny, nz, data: vec![val; nx * ny * nz],};
        f
    }
    
    fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        iz * self.nx * self.ny + iy * self.nx + ix
    }

    fn get(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        self.data[self.idx(ix, iy, iz)]
    }

    fn set(&mut self, ix: usize, iy: usize, iz: usize, val: f64) {
        let idx = self.idx(ix, iy, iz);
        self.data[idx] = val;
    }

    fn elementwise_inplace_add(&mut self, other: &ThreeDField) {
        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x += y;
        }
    }

    fn elementwise_inplace_sub(&mut self, other: &ThreeDField) {
        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x -= y;
        }
    }

    fn elementwise_inplace_mult(&mut self, other: &ThreeDField) {
        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x *= y;
        }
    }

    fn elementwise_inplace_div(&mut self, other: &ThreeDField) {
        for (x,y) in self.data.iter_mut().zip(&other.data) {
            *x /= y;
        }
    }

    fn scalar_inplace_add(&mut self, s: f64) {
        for x in self.data.iter_mut() {
            *x += s;
        }
    }

    fn scalar_inplace_mult(&mut self, s: f64) {
        for x in self.data.iter_mut() {
            *x *= s;
        }
    }
    
}


