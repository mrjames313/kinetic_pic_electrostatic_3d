use anyhow::Result;
use glam::DVec3;  // might consider using ndarray in the future
use std::path::Path;
use vtkio::model::{Attribute, Attributes, ByteOrder, DataSet, Extent,
                   ImageDataPiece, Piece, Version, Vtk};

// internal 
use super::ThreeDField;
use crate::constants::*;
use crate::particles::Species;

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
    // Maybe there should be a part of this spec that is open / accessible - basically
    // corresponding to the grid itself and basic properties of it.  Wrap that up in a
    // struct?
    
    x_dim: SingleDimSpec,
    y_dim: SingleDimSpec,
    z_dim: SingleDimSpec,
    // TODO: this indicates bad design to me - need to think about ownership and access
    // between world and species
    pub node_volume: ThreeDField<f64>,

    
    phi: ThreeDField<f64>,
    rho: ThreeDField<f64>,
    ef: ThreeDField<DVec3>,

}

impl ThreeDWorldSpec {
    
    pub fn init(x_dim: SingleDimSpec, y_dim: SingleDimSpec, z_dim: SingleDimSpec) -> Result<Self> {
        let phi = ThreeDField::init(x_dim.n, y_dim.n, z_dim.n, 0.0);
        let rho = ThreeDField::init(x_dim.n, y_dim.n, z_dim.n, 0.0);
        let ef = ThreeDField::init(x_dim.n, y_dim.n, z_dim.n, DVec3::new(0.0, 0.0, 0.0));
        let node_volume = ThreeDField::init(x_dim.n, y_dim.n, z_dim.n, 0.0);
        
        let mut spec = Self{x_dim:x_dim, y_dim:y_dim, z_dim:z_dim,
                            phi:phi, rho:rho, ef:ef, node_volume};
        spec.set_node_volumes();
        
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

    // should add bounds checking?  Otherwise could get negative or out-of-bounds
    pub fn get_full_node_index(&self, real_coord : DVec3) -> DVec3 {
        let mut index : DVec3 = [0.0, 0.0, 0.0].into();
        index[0] = (real_coord[0] - self.x_dim.min) / self.x_dim.delta;
        index[1] = (real_coord[1] - self.y_dim.min) / self.y_dim.delta;
        index[2] = (real_coord[2] - self.z_dim.min) / self.z_dim.delta;
        index
    }
    
    // Could probably optimize this by precomputing powers and matching on count, but
    // only called at startup, so maybe not worth it.
    fn set_node_volumes(&mut self) {
        let vol = self.x_dim.delta * self.y_dim.delta * self.z_dim.delta;
        let half : f64 = 0.5;
        for i in 0 .. self.x_dim.n {
            for j in 0 .. self.y_dim.n {
                for k in 0 .. self.z_dim.n {
                    let count = (i == 0 || i == self.x_dim.n - 1) as usize +
                        (j == 0 || j == self.y_dim.n - 1) as usize +
                        (k == 0 || k == self.z_dim.n - 1) as usize;

                    self.node_volume.set(i,j,k, vol * half.powf(count as f64));
                }
            }
        }
    }


    pub fn set_phi(&mut self, i: usize, j: usize, k: usize, val: f64) {
        self.phi.set(i,j,k,val);
    }

    // Requires that these fields are the same dim, etc
    pub fn compute_rho(&mut self, species : &Vec<Species>) {
        self.rho.set_all(0.0);
        for s in species.iter() {
            self.rho.elementwise_inplace_add_scaled(s.charge, &s.number_density);
        }
    }
    
    pub fn get_ef(&mut self, i: usize, j: usize, k: usize) -> DVec3 {
        self.ef.get(i,j,k)
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
//                    if i==5 && j==6 {
//                        println!("Ef at {i}, {j}, {k} is {ef}");
//                    }
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

    // TODO: start thinking about how to incorporate species into the VTI output
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
    

// Probably should set up different numbers of cells, deltas, and initial positions
// for the world, in both functions below
#[test]
fn efield_of_constant_phi_is_zero() -> anyhow::Result<()> {
    let x_dim = match SingleDimSpec::init(21, -0.1, 0.1) {
        Ok(s) => s,
        Err(_) => {return Err(anyhow::anyhow!("bad 3d spec for x"));}
    };

    let y_dim = match SingleDimSpec::init(21, -0.1, 0.1) {
        Ok(s) => s,
        Err(_) => {return Err(anyhow::anyhow!("bad 3d spec for y"));}
    };

    let z_dim = match SingleDimSpec::init(21, -0.0, 0.2) {
        Ok(s) => s,
        Err(_) => {return Err(anyhow::anyhow!("bad 3d spec for z"));}
    };

    let mut world = match ThreeDWorldSpec::init(x_dim, y_dim, z_dim) {
        Ok(s) => s,
        Err(_) => {
            println!("Failed to create a three d world spec");
            return Err(anyhow::anyhow!("bad world spec"));
        }
    };
    let const_val: f64 = 13.444;
    
    for i in 0..world.get_x_dim_n() {
        for j in 0..world.get_y_dim_n() {
            for k in 0..world.get_z_dim_n() {
                world.set_phi(i, j, k, const_val);
            }
        }
    }
    world.compute_ef();
    
    let tol = 1e-12;
    for i in 0..world.get_x_dim_n() {
        for j in 0..world.get_y_dim_n() {
            for k in 0..world.get_z_dim_n() {
                let e = world.get_ef(i,j,k);
                assert!(e[0].abs() < tol);
                assert!(e[1].abs() < tol);
                assert!(e[2].abs() < tol);
            }
        }
    }
    Ok(())
}

#[test]
fn efield_of_linear_phi_is_constant() -> anyhow::Result<()> {
    let x_dim = match SingleDimSpec::init(21, -0.1, 0.1) {
        Ok(s) => s,
        Err(_) => {return Err(anyhow::anyhow!("bad 3d spec for x"));}
    };

    let y_dim = match SingleDimSpec::init(21, -0.1, 0.1) {
        Ok(s) => s,
        Err(_) => {return Err(anyhow::anyhow!("bad 3d spec for y"));}
    };

    let z_dim = match SingleDimSpec::init(21, -0.0, 0.2) {
        Ok(s) => s,
        Err(_) => {return Err(anyhow::anyhow!("bad 3d spec for z"));}
    };

    let mut world = match ThreeDWorldSpec::init(x_dim, y_dim, z_dim) {
        Ok(s) => s,
        Err(_) => {
            println!("Failed to create a three d world spec");
            return Err(anyhow::anyhow!("bad world spec"));
        }
    };

    let (x0, y0, z0) = (0.0, 0.0, 0.0);
    let (a, b, c) = (1.7, -0.4, 0.9);
    let (dx, dy, dz) = (x_dim.delta, y_dim.delta, z_dim.delta);

    fn linear_interp(i: usize, init: f64, scale: f64, delta: f64) ->
        f64 {init + (i as f64) * scale * delta }
    
    for i in 0..world.get_x_dim_n() {
        for j in 0..world.get_y_dim_n() {
            for k in 0..world.get_z_dim_n() {
                let x = linear_interp(i, x0, a, dx);
                let y = linear_interp(j, y0, b, dy);
                let z = linear_interp(k, z0, c, dz);
                world.set_phi(i, j, k, x + y + z);
            }
        }
    }
    world.compute_ef();
    
    let tol = 1e-8;
    for i in 0..world.get_x_dim_n() {
        for j in 0..world.get_y_dim_n() {
            for k in 0..world.get_z_dim_n() {
                let e = world.get_ef(i,j,k);
                assert!((e[0] + a).abs() < tol, "values {} and {} should only differ in sign", e[0], a );
                assert!((e[1] + b).abs() < tol, "values {} and {} should only differ in sign", e[1], b);
                assert!((e[2] + c).abs() < tol, "values {} and {} should only differ in sign", e[2], c );
            }
        }
    }
    Ok(())
}
