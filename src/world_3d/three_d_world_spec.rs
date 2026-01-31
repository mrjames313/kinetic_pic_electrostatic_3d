use anyhow::Result;
use glam::DVec3;  // might consider using ndarray in the future
use std::path::Path;
use vtkio::model::{Attribute, Attributes, ByteOrder, DataSet, Extent,
                   ImageDataPiece, Piece, Version, Vtk};

// internal 
use super::ThreeDField;
use crate::constants::*;

#[derive(Copy, Clone, Debug)]
pub struct SingleDimSpec {
    pub n: usize,
    pub min: f64,
    pub max: f64,
    // computed fields
    pub delta: f64,
    pub center: f64,
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

    // some helper functions for key locations in the world
    pub fn get_min_corner(&self) -> DVec3 {
        DVec3::new(self.x_dim.min, self.y_dim.min, self.z_dim.min)
    }

    pub fn get_center(&self) -> DVec3 {
        DVec3::new(self.x_dim.center, self.y_dim.center, self.z_dim.center)
    }

    pub fn get_max_corner(&self) -> DVec3 {
        DVec3::new(self.x_dim.max, self.y_dim.max, self.z_dim.max)
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

    pub fn compute_ef(&mut self) -> Result<(), String> {
        let two_dx : f64 = 2.0 * self.x_dim.delta;
        let two_dy : f64 = 2.0 * self.y_dim.delta;
        let two_dz : f64 = 2.0 * self.z_dim.delta;

        let mut ef: DVec3 = [0.0, 0.0, 0.0].into();
        
        for i in 0..self.x_dim.n {
            for j in 0..self.y_dim.n {
                for k in 0..self.z_dim.n {
                    // In each inner loop iteration, set each component of ef, then
                    // at the end, assign it to self.ef[i,j,k]
                    if i == 0 {
                        ef[0] = - (-3.0 * self.phi.get(i,j,k)
                                   + 4.0 * self.phi.get(i+1, j, k)
                                   - self.phi.get(i+2, j, k))
                            / two_dx;
                    } else if i == self.x_dim.n - 1 {
                        ef[0] = - (3.0 * self.phi.get(i,j,k)
                                   - 4.0 * self.phi.get(i-1, j, k)
                                   + self.phi.get(i-2, j, k))
                            / two_dx;
                    } else {
                        ef[0] = -(self.phi.get(i+1,j,k) - self.phi.get(i-1,j,k))
                            / two_dx;
                    }
                    
                    if j == 0 {
                        ef[1] = - (-3.0 * self.phi.get(i,j,k)
                                   + 4.0 * self.phi.get(i, j+1, k)
                                   - self.phi.get(i, j+2, k))
                            / two_dy;
                    } else if j == self.y_dim.n - 1 {
                        ef[1] = - (3.0 * self.phi.get(i,j,k)
                                   - 4.0 * self.phi.get(i, j-1, k)
                                   + self.phi.get(i, j-2, k))
                            / two_dy;
                    } else {
                        ef[1] = -(self.phi.get(i,j+1,k) - self.phi.get(i,j-1,k))
                            / two_dy;
                    }

                    if k == 0 {
                        ef[2] = - (-3.0 * self.phi.get(i,j,k)
                                   + 4.0 * self.phi.get(i, j, k+1)
                                   - self.phi.get(i, j, k+2))
                            / two_dz;
                    } else if k == self.z_dim.n - 1 {
                        ef[2] = - (3.0 * self.phi.get(i,j,k)
                                   - 4.0 * self.phi.get(i, j, k-1)
                                   + self.phi.get(i, j, k-2))
                            / two_dz;
                    } else {
                        ef[2] = -(self.phi.get(i,j,k+1) - self.phi.get(i,j,k-1))
                            / two_dz;
                    }

                    self.ef.set(i,j,k, ef);
                    if i==5 && j==6 {
                        println!("Ef at {i}, {j}, {k} is {ef}");
                    }
                }
            }
        }
        
        Ok(())
    }

    
    fn flatten_dvec3(vs: &[DVec3]) -> Vec<f64> {
        let mut out = Vec::with_capacity(3 * vs.len());
        for v in vs {
            out.push(v.x);
            out.push(v.y);
            out.push(v.z);
        }
        out
    }
    
    pub fn write_world_vti(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
//        ef_xyz: &[f64], // flattened as [ex0,ey0,ez0, ex1,ey1,ez1, ...]
        let npts = self.x_dim.n * self.y_dim.n * self.z_dim.n;
        anyhow::ensure!(self.phi.len() == npts, "phi wrong length");
        anyhow::ensure!(self.rho.len() == npts, "rho wrong length");
        anyhow::ensure!(self.ef.len() == npts, "ef_xyz wrong length (need 3*npts)"); // this might not work 

        // Attach arrays as POINT_DATA (common for potentials/fields sampled at grid points).
        let mut attrs = Attributes::new();
        attrs.point.push(Attribute::scalars("phi", 1).with_data(self.phi.data().to_vec() ));
        attrs.point.push(Attribute::scalars("rho", 1).with_data(self.rho.data().to_vec() ));
        let ef_flat = Self::flatten_dvec3(self.ef.data());
        attrs.point.push(Attribute::vectors("ef").with_data(ef_flat));

        // ImageData uses an extent + origin + spacing. Extent::Dims is the legacy “dims” form. :contentReference[oaicite:5]{index=5}
        // ImageData uses inclusive ranges, so 0..20 includes 21 points.
        let extent = Extent::Ranges([0..=((self.x_dim.n as i32) - 1), 0..=((self.y_dim.n as i32) - 1), 0..=((self.z_dim.n as i32) - 1) ]);

        let piece = ImageDataPiece {
            extent: extent.clone(),
            data: attrs,
        };

        let vtk = Vtk {
            version: Version::new((2, 3)),
            byte_order: ByteOrder::LittleEndian,
            title: "ThreeDWorldSpec snapshot".to_string(),
            file_path: None,
            data: DataSet::ImageData {
                extent,
                origin: [self.x_dim.min as f32, self.y_dim.min as f32, self.z_dim.min as f32],
                spacing: [self.x_dim.delta as f32, self.y_dim.delta as f32, self.z_dim.delta as f32],
                meta: None,
                pieces: vec![Piece::Inline(Box::new(piece))],
            },
        };

        vtk.export(path)?;
        Ok(())
    }

    //pub fn compute_ef() -> Result<()> {
   // }
}
    
