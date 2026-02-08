use anyhow::Result;
use glam::DVec3;  // might consider using ndarray in the future
use std::path::Path;
use std::time::Instant;
use vtkio::model::{Attribute, Attributes, ByteOrder, DataSet, Extent,
                   ImageDataPiece, Piece, Version, Vtk};

// internal 
use super::ThreeDField;
use crate::constants::*;
use crate::particles::Species;
use crate::output::TimeInfo;
use crate::output::IterInfo;


// TODO: refine this, probably have a read-only public interface to all 3 specs
#[derive(Copy, Clone, Debug)]
pub struct SingleDimSpec {
    n: usize,
    min: f64,
    max: f64,
    // computed fields
    delta: f64,
    center: f64,
}

impl SingleDimSpec {
    pub fn new(n: usize, min: f64, max: f64) -> anyhow::Result<Self> {
        assert!(n >= 2, "n must be >= 2");
        assert!(max > min, "max must be > min");
        
        Ok( Self{n:n, min:min, max:max,
                 delta:(max - min)/(n-1) as f64,
                 center:(max + min) / 2.0 } )
    }

    pub fn n(&self) -> usize { self.n }
    pub fn min(&self) -> f64 { self.min }
    pub fn max(&self) -> f64 { self.max }
    pub fn delta(&self) -> f64 { self.delta }
    pub fn center(&self) -> f64 { self.center }
    
    pub fn print(&self) {
        println!("Extent: [{:.4}, {:.4}], {} cells, delta {}, center{}", self.min,
                 self.max, self.n-1, self.delta, self.center);
    }
}

// Iteration and time are assumed to start at 0 and 0.0
pub struct TimeRepresentation {
    iteration: usize,
    sim_time: f64,
    wall_time: f64, // TODO: finish this
    wall_timer: Instant,
    dt: f64, // fixed
}

impl TimeRepresentation {
    pub fn iteration(&self) -> usize { self.iteration }
    pub fn sim_time(&self) -> f64 { self.sim_time }
    pub fn wall_time(&self) -> f64 { self.wall_time }
    pub fn dt(&self) -> f64 { self.dt }

    pub fn start_iteration_time(&mut self) {
        self.iteration = 0;
        self.sim_time = 0.0;
        self.wall_time = 0.0;
        self.wall_timer = Instant::now();
    }

    pub fn advance_iteration(&mut self) {
        self.wall_time = self.wall_timer.elapsed().as_secs_f64();
        self.iteration += 1;
        self.sim_time += self.dt;
    }

}

pub struct ThreeDWorldSpec {
    x_dim: SingleDimSpec,
    y_dim: SingleDimSpec,
    z_dim: SingleDimSpec,
    node_volume: ThreeDField<f64>,
}

impl ThreeDWorldSpec {
    pub fn new(x_dim: SingleDimSpec, y_dim: SingleDimSpec, z_dim: SingleDimSpec)
               -> anyhow::Result<Self> {
        // Node volumes are reduced along faces, edges, corners
        let vol = x_dim.delta * y_dim.delta * z_dim.delta;
        let mut node_volume = ThreeDField::new(x_dim.n, y_dim.n, z_dim.n, vol)?;

        let half : f64 = 0.5;
        for i in 0 .. x_dim.n() {
            for j in 0 .. y_dim.n() {
                for k in 0 .. z_dim.n() {
                    let count = (i == 0 || i == x_dim.n() - 1) as usize +
                        (j == 0 || j == y_dim.n() - 1) as usize +
                        (k == 0 || k == z_dim.n() - 1) as usize;
                    if count > 0 { // internal nodes already set
                        node_volume.set(i,j,k, vol * half.powf(count as f64));
                    }
                }
            }
        }
        Ok( Self { x_dim, y_dim, z_dim, node_volume } )
    }

    pub fn x_dim(&self) -> &SingleDimSpec { &self.x_dim }
    pub fn y_dim(&self) -> &SingleDimSpec { &self.y_dim }
    pub fn z_dim(&self) -> &SingleDimSpec { &self.z_dim }
    pub fn node_volume(&self) -> &ThreeDField<f64> { &self.node_volume }

    // some helper functions for key locations in the world
    pub fn get_min_corner(&self) -> DVec3 {
        DVec3::new(self.x_dim.min(), self.y_dim.min(),
                   self.z_dim.min())
    }

    pub fn get_center(&self) -> DVec3 {
        DVec3::new(self.x_dim.center(), self.y_dim.center(),
                   self.z_dim.center())
    }

    pub fn get_max_corner(&self) -> DVec3 {
        DVec3::new(self.x_dim.max(), self.y_dim.max(),
                   self.z_dim.max())
    }

    // TODO: determine if this assert is too expensive, maybe make debug-only? debug_assert!
    pub fn get_full_node_index(&self, real_coord : DVec3) -> DVec3 {
        assert!(real_coord[0] >= self.x_dim.min() && real_coord[0] <= self.x_dim.max() &&
               real_coord[1] >= self.y_dim.min() && real_coord[1] <= self.y_dim.max() &&
               real_coord[2] >= self.z_dim.min() && real_coord[2] <= self.z_dim.max(),
               "One of the node coordinates {real_coord} is out of bounds");
        
        let mut index : DVec3 = [0.0, 0.0, 0.0].into();
        index[0] = (real_coord[0] - self.x_dim.min()) / self.x_dim.delta();
        index[1] = (real_coord[1] - self.y_dim.min()) / self.y_dim.delta();
        index[2] = (real_coord[2] - self.z_dim.min()) / self.z_dim.delta();
        index
    }

    pub fn get_full_node_index_no_assert(&self, real_coord : DVec3) -> DVec3 {
        let mut index : DVec3 = [0.0, 0.0, 0.0].into();
                index[0] = (real_coord[0] - self.x_dim.min()) / self.x_dim.delta();
        index[1] = (real_coord[1] - self.y_dim.min()) / self.y_dim.delta();
        index[2] = (real_coord[2] - self.z_dim.min()) / self.z_dim.delta();
        index
    }

}

pub struct ThreeDWorld {
    world_spec: ThreeDWorldSpec,
    time: TimeRepresentation,
    
    phi: ThreeDField<f64>,
    rho: ThreeDField<f64>,
    ef: ThreeDField<DVec3>,
}

// Diagnostic info
// TODO: this is just placeholder now, need to compute potential energy,
// and total e
pub fn get_iter_info_from_world(_world: &ThreeDWorld) -> IterInfo {
    let potential_e = 0.0;
    let total_e = 0.0;
    IterInfo {potential_e: potential_e, total_e: total_e}
}

pub fn get_time_info_from_world(world: &ThreeDWorld) -> TimeInfo {
    let iteration = world.time().iteration();
    let sim_time = world.time().sim_time();
    let wall_time = world.time().wall_time();
    TimeInfo {iteration: iteration, sim_time: sim_time, wall_time: wall_time}
}
    

impl ThreeDWorld {

    pub fn new(world_spec: ThreeDWorldSpec, dt: f64) -> anyhow::Result<Self> {
        let nx = world_spec.x_dim().n();
        let ny = world_spec.y_dim().n();
        let nz = world_spec.z_dim().n();
        
        Ok(Self {
            world_spec,
            time: TimeRepresentation{iteration: 0, sim_time: 0.0, wall_time: 0.0,
                                     wall_timer: Instant::now(), dt: dt},
            phi: ThreeDField::new(nx, ny, nz, 0.0)?,
            rho: ThreeDField::new(nx, ny, nz, 0.0)?,
            ef: ThreeDField::new(nx, ny, nz, DVec3::new(0.0, 0.0, 0.0))?,
        } )
    }

    pub fn world_spec(&self) -> &ThreeDWorldSpec {&self.world_spec }
    pub fn time(&self) -> &TimeRepresentation {&self.time }
    pub fn mut_time(&mut self) -> &mut TimeRepresentation {&mut self.time }
    
    pub fn phi(&self) -> &ThreeDField<f64> {&self.phi}
    pub fn rho(&self) -> &ThreeDField<f64> {&self.rho}
    pub fn ef(&self) -> &ThreeDField<DVec3> {&self.ef}
    
    pub fn set_phi(&mut self, i: usize, j: usize, k: usize, val: f64) {
        self.phi.set(i,j,k,val);
    }

    // Requires that these fields are the same dim, etc
    pub fn compute_rho(&mut self, species : &Vec<Species>) {
        self.rho.set_all(0.0);
//        // debugging indices
//        let debug_indices: Vec<[usize; 3]> = [[5,5,5], [5,5,15], [5,15,5], [5,15,15],
//                                              [15,5,5], [15,5,15], [15,15,5], [15,15,15]].into();
//        // debug code
//        // TODO: figure out if we need to put testing around this
//        for arr in debug_indices.iter() {
//            println!("At index [{}, {}, {}] have value {}", arr[0], arr[1], arr[2],
//                     self.rho.get(arr[0], arr[1], arr[2]));
//        }
        

        for s in species.iter() {
//            println!("New species {}, charge {}", s.name, s.charge);
            
            self.rho.elementwise_inplace_add_scaled(s.charge, &s.number_density);
//            // debug code
//            for arr in debug_indices.iter() {
//                println!("At index [{}, {}, {}] have node volume {}", arr[0], arr[1], arr[2],
//                         self.node_volume.get(arr[0], arr[1], arr[2]));
//
//                println!("At index [{}, {}, {}] have number_density {}", arr[0], arr[1], arr[2],
//                         s.number_density.get(arr[0], arr[1], arr[2]));
//
//                println!("At index [{}, {}, {}] have rho {}", arr[0], arr[1], arr[2],
//                         self.rho.get(arr[0], arr[1], arr[2]));
//            }

        }
    }
    
    pub fn get_ef(&mut self, i: usize, j: usize, k: usize) -> DVec3 {
        self.ef.get(i,j,k)
    }

    pub fn interpolate_ef(&self, full_idx: DVec3) -> DVec3 {
        self.ef.linear_interpolate(full_idx)
    }
    
    pub fn print(&self) -> Result <()> {
        println!("Three dimensional world mesh with dimensions:");
        print!("X: ");
        self.world_spec.x_dim().print();
        print!("Y: ");
        self.world_spec.y_dim().print();
        print!("Z: ");
        self.world_spec.z_dim().print();
        Ok(())
    }

    pub fn solve_potential_gs_sor(&mut self, max_iter : usize) -> Result<usize, String> {
        // params that control execution
        let w : f64 = 1.4;
        let l2_conv : f64 = 1e-6;

        // precompute some commonly used values
        let inv_dx2 : f64 = 1.0 / (self.world_spec.x_dim().delta() * self.world_spec.x_dim().delta());
        let inv_dy2 : f64 = 1.0 / (self.world_spec.y_dim().delta() * self.world_spec.y_dim().delta());
        let inv_dz2 : f64 = 1.0 / (self.world_spec.z_dim().delta() * self.world_spec.z_dim().delta());

        let mut l2 : f64 = 1e12;
        let found = {
            let mut result = None;

            for iter in 0..max_iter {
                for i in 1..self.world_spec.x_dim().n() - 1 {
                    for j in 1..self.world_spec.y_dim().n() - 1 {
                        for k in 1..self.world_spec.z_dim().n() - 1 {
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
                    for i in 1..self.world_spec.x_dim().n() - 1 {
                        for j in 1..self.world_spec.y_dim().n() - 1 {
                            for k in 1..self.world_spec.z_dim().n() - 1 {
                                let r = self.rho.get(i,j,k) / EPS0 -
                                    self.phi.get(i,j,k) * (2.0 * inv_dx2 + 2.0 * inv_dy2 + 2.0 * inv_dz2) +
                                    inv_dx2 * (self.phi.get(i-1,j,k) + self.phi.get(i+1,j,k)) +
                                    inv_dy2 * (self.phi.get(i,j-1,k) + self.phi.get(i,j+1,k)) +
                                    inv_dz2 * (self.phi.get(i,j,k-1) + self.phi.get(i,j,k+1));
                                sum += r * r;
                            }
                        }
                    }
                    l2 = (sum / (self.world_spec.x_dim().n() * self.world_spec.y_dim().n() * self.world_spec.z_dim().n()) as f64).sqrt();
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
        let two_dx : f64 = 2.0 * self.world_spec.x_dim().delta();
        let two_dy : f64 = 2.0 * self.world_spec.y_dim().delta();
        let two_dz : f64 = 2.0 * self.world_spec.z_dim().delta();

        let mut ef: DVec3 = [0.0, 0.0, 0.0].into();
        
        for i in 0..self.world_spec.x_dim().n() {
            for j in 0..self.world_spec.y_dim().n() {
                for k in 0..self.world_spec.z_dim().n() {
                    // In each inner loop iteration, set each component of ef, then
                    // at the end, assign it to self.ef[i,j,k]
                    if i == 0 {
                        ef[0] = - (-3.0 * self.phi.get(i,j,k)
                                   + 4.0 * self.phi.get(i+1, j, k)
                                   - self.phi.get(i+2, j, k))
                            / two_dx;
                    } else if i == self.world_spec.x_dim().n() - 1 {
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
                    } else if j == self.world_spec.y_dim().n() - 1 {
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
                    } else if k == self.world_spec.z_dim().n() - 1 {
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

    pub fn compute_potential_energy(&self) -> f64 {
        let mut pe: f64 = 0.0;
        for i in 0..self.world_spec.x_dim().n() {
            for j in 0..self.world_spec.y_dim().n() {
                for k in 0..self.world_spec.z_dim().n() {
                    pe += self.ef.get(i,j,k).length_squared() * self.world_spec.node_volume().get(i,j,k);
                }
            }
        }
        pe *= 0.5 * EPS0;
        pe
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
        let npts = self.world_spec.x_dim().n() * self.world_spec.y_dim().n() * self.world_spec.z_dim().n();
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
        let extent = Extent::Ranges([0..=((self.world_spec.x_dim().n() as i32) - 1), 0..=((self.world_spec.y_dim().n() as i32) - 1), 0..=((self.world_spec.z_dim().n() as i32) - 1) ]);

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
                origin: [self.world_spec.x_dim().min() as f32, self.world_spec.y_dim().min() as f32, self.world_spec.z_dim().min() as f32],
                spacing: [self.world_spec.x_dim().delta() as f32, self.world_spec.y_dim().delta() as f32, self.world_spec.z_dim().delta() as f32],
                meta: None,
                pieces: vec![Piece::Inline(Box::new(piece))],
            },
        };

        vtk.export(path)?;
        Ok(())
    }
}
    

// Probably should set up different numbers of cells, deltas, and initial positions
// for the world, in both functions below
#[test]
fn efield_of_constant_phi_is_zero() -> anyhow::Result<()> {
    let x_dim = SingleDimSpec::new(21, -0.1, 0.1)?;
    let y_dim = SingleDimSpec::new(21, -0.1, 0.1)?;
    let z_dim = SingleDimSpec::new(21, -0.0, 0.2)?;
    let world_spec = ThreeDWorldSpec::new(x_dim, y_dim, z_dim)?;
    let dt: f64 = 2e-10;
    let mut world = ThreeDWorld::new(world_spec, dt)?;

    let const_val: f64 = 13.444;
    
    for i in 0..world.world_spec().x_dim().n() {
        for j in 0..world.world_spec().y_dim().n() {
            for k in 0..world.world_spec().z_dim().n() {
                world.set_phi(i, j, k, const_val);
            }
        }
    }
    if let Err(e) = world.compute_ef() {
        return Err(anyhow::anyhow!("EF computation failed with error {}", e))
    }

    let tol = 1e-12;
    for i in 0..world.world_spec().x_dim().n() {
        for j in 0..world.world_spec().y_dim().n() {
            for k in 0..world.world_spec().z_dim().n() {
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
    let x_dim = SingleDimSpec::new(21, -0.1, 0.1)?;
    let y_dim = SingleDimSpec::new(21, -0.1, 0.1)?;
    let z_dim = SingleDimSpec::new(21, -0.0, 0.2)?;
    let world_spec = ThreeDWorldSpec::new(x_dim, y_dim, z_dim)?;
    let dt: f64 = 2e-10;
    let mut world = ThreeDWorld::new(world_spec, dt)?;

    let (x0, y0, z0) = (0.0, 0.0, 0.0);
    let (a, b, c) = (1.7, -0.4, 0.9);
    let (dx, dy, dz) = (x_dim.delta(), y_dim.delta(), z_dim.delta());

    fn linear_interp(i: usize, init: f64, scale: f64, delta: f64) ->
        f64 {init + (i as f64) * scale * delta }
    
    for i in 0..world.world_spec().x_dim().n() {
        for j in 0..world.world_spec().y_dim().n() {
            for k in 0..world.world_spec().z_dim().n() {
                let x = linear_interp(i, x0, a, dx);
                let y = linear_interp(j, y0, b, dy);
                let z = linear_interp(k, z0, c, dz);
                world.set_phi(i, j, k, x + y + z);
            }
        }
    }
    if let Err(e) = world.compute_ef() {
        return Err(anyhow::anyhow!("EF computation failed with error {}", e))
    }
    
    let tol = 1e-8;
    for i in 0..world.world_spec().x_dim().n() {
        for j in 0..world.world_spec().y_dim().n() {
            for k in 0..world.world_spec().z_dim().n() {
                let e = world.get_ef(i,j,k);
                assert!((e[0] + a).abs() < tol, "values {} and {} should only differ in sign", e[0], a );
                assert!((e[1] + b).abs() < tol, "values {} and {} should only differ in sign", e[1], b);
                assert!((e[2] + c).abs() < tol, "values {} and {} should only differ in sign", e[2], c );
            }
        }
    }
    Ok(())
}
