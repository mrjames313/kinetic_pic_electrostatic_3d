use anyhow::Result;
use glam::DVec3;  // might consider using ndarray in the future

// internal 
use super::ThreeDField;
use crate::constants::*;


pub struct SingleDimSpec {
    n: usize,
    min: f64,
    max: f64,
    // computed fields
    delta: f64,
    center: f64,
}

impl SingleDimSpec {
    pub fn init(n: usize, min: f64, max: f64) -> Result<Self> {
        let spec = Self{n:n, min:min, max:max,
                            delta:(max - min)/(n-1) as f64,
                            center:(max - min) / 2.0 };
        Ok(spec)
    }

    pub fn print(&self) -> Result <()> {
        println!("Extent: [{:.4}, {:.4}], {} cells", self.min, self.max, self.n-1);
        Ok(())
    }
}

pub struct ThreeDWorldSpec {
    x_dim: SingleDimSpec,
    y_dim: SingleDimSpec,
    z_dim: SingleDimSpec,
    phi: ThreeDField<f64>,
    rho: ThreeDField<f64>,
    ef: ThreeDField<DVec3>,
}

impl ThreeDWorldSpec {
    
    pub fn init(x_dim: SingleDimSpec, y_dim: SingleDimSpec, z_dim: SingleDimSpec) -> Result<Self> {
        let phi = ThreeDField::init(x_dim.n, y_dim.n, z_dim.n, 0.0);
        let rho = ThreeDField::init(x_dim.n, y_dim.n, z_dim.n, 0.0);
        let ef = ThreeDField::init(x_dim.n, y_dim.n, z_dim.n, DVec3::new(0.0, 0.0, 0.0));
        let spec = Self{x_dim:x_dim, y_dim:y_dim, z_dim:z_dim,
                        phi:phi, rho:rho, ef:ef};
        Ok(spec)
    }

    pub fn get_x_dim_n(&self) -> usize {
        self.x_dim.n
    }

    pub fn get_y_dim_n(&self) -> usize {
        self.y_dim.n
    }

    pub fn get_z_dim_n(&self) -> usize {
        self.z_dim.n
    }

    pub fn set_phi(&mut self, i: usize, j: usize, k: usize, val: f64) {
        self.phi.set(i,j,k,val);
    }


    pub fn print(&self) -> Result <()> {
        println!("Three dimensional world mesh with dimensions:");
        print!("X: ");
        self.x_dim.print()?;
        print!("Y: ");
        self.y_dim.print()?;
        print!("Z: ");
        self.z_dim.print()?;
        Ok(())
    }

    pub fn solve_potential_gs_sor(&mut self, max_iter : usize) -> Result<usize, String> {
        // params that control execution
        let w : f64 = 1.4;
        let l2_conv : f64 = 1e-6;

        // precompute some commonly used values
        let inv_dx2 : f64 = 1.0 / (self.x_dim.delta * self.x_dim.delta);
        let inv_dy2 : f64 = 1.0 / (self.y_dim.delta * self.y_dim.delta);
        let inv_dz2 : f64 = 1.0 / (self.z_dim.delta * self.z_dim.delta);

        let mut l2 : f64 = 1e12;
        let found = {
            let mut result = None;

            for iter in 0..max_iter {
                for i in 1..self.x_dim.n - 1 {
                    for j in 1..self.y_dim.n - 1 {
                        for k in 1..self.z_dim.n - 1 {
                            let phi_new = ( self.rho.get(i,j,k) / EPS0 +
                                            inv_dx2 * (self.phi.get(i-1,j,k) + self.phi.get(i+1,j,k)) +
                                            inv_dy2 * (self.phi.get(i,j-1,k) + self.phi.get(i,j+1,k)) +
                                            inv_dz2 * (self.phi.get(i,j,k-1) + self.phi.get(i,j,k+1)) ) /
                                (2.0 * inv_dx2 + 2.0 * inv_dy2 + 2.0 * inv_dz2);
                            let phi_update = (1.0 - w) * self.phi.get(i,j,k) + w * phi_new;
                            self.phi.set(i,j,k, phi_update);
                        }
                    }
                }

                // Periodic check for convergence
                if iter % 50 == 0 {
                    let mut sum: f64 = 0.0;
                    for i in 1..self.x_dim.n - 1 {
                        for j in 1..self.y_dim.n - 1 {
                            for k in 1..self.z_dim.n - 1 {
                                let r = self.rho.get(i,j,k) / EPS0 -
                                    self.phi.get(i,j,k) * (2.0 * inv_dx2 + 2.0 * inv_dy2 + 2.0 * inv_dz2) +
                                    inv_dx2 * (self.phi.get(i-1,j,k) + self.phi.get(i+1,j,k)) +
                                    inv_dy2 * (self.phi.get(i,j-1,k) + self.phi.get(i,j+1,k)) +
                                    inv_dz2 * (self.phi.get(i,j,k-1) + self.phi.get(i,j,k+1));
                                sum += r * r;
                            }
                        }
                    }
                    l2 = (sum / (self.x_dim.n * self.y_dim.n * self.z_dim.n) as f64).sqrt();
                    if l2 < l2_conv {
                        result = Some(iter);
                        break;
                    }
                }

            }
            result
        };
        match found {
            Some(i) => { Ok(i) }
            None => {
                Err(format!("GS SOR didn't converge.  L2 residual {l2:.6}"))
            }
        }
    }

    //pub fn compute_ef() -> Result<()> {
   // }
}
    
