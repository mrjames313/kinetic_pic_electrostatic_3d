use anyhow::Result;
use glam::DVec3;  // might consider using ndarray in the future
use std::fmt;
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
    pub fn new(n: usize, min: f64, max: f64) -> Self {
        assert!(n >= 2, "n must be >= 2");
        assert!(max > min, "max must be > min");
        
        Self{n:n, min:min, max:max,
             delta:(max - min)/(n-1) as f64,
             center:(max + min) / 2.0 }
    }

    pub fn n(&self) -> usize { self.n }
    pub fn min(&self) -> f64 { self.min }
    pub fn max(&self) -> f64 { self.max }
    pub fn delta(&self) -> f64 { self.delta }
    pub fn center(&self) -> f64 { self.center }
    
}

impl fmt::Display for SingleDimSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Extent: [{:.4}, {:.4}], {} cells, delta {}, center {}",
            self.min,
            self.max,
            self.n - 1,
            self.delta,
            self.center
        )
    }
}


// Iteration and time are assumed to start at 0 and 0.0
pub struct TimeRepresentation {
    iteration: usize,
    sim_time: f64,
    wall_time: f64,
    wall_timer: Instant,
    dt: f64,
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

#[derive(Copy, Clone, Debug)]
pub struct SorSolverConfig {
    pub omega: f64,        // relaxation factor
    pub l2_conv: f64,      // convergence threshold
    pub check_every: usize // how often to compute residual
}

impl Default for SorSolverConfig {
    fn default() -> Self {
        Self {
            omega: 1.4,
            l2_conv: 1e-6,
            check_every: 50,
        }
    }
}

pub struct ThreeDWorldSpec {
    x_dim: SingleDimSpec,
    y_dim: SingleDimSpec,
    z_dim: SingleDimSpec,
    node_volume: ThreeDField<f64>,
}

impl ThreeDWorldSpec {

    #[inline(always)]
    fn check_dims_basic(&self) {
        // cheap invariants; useful everywhere
        debug_assert!(self.x_dim.n() >= 2 && self.y_dim.n() >= 2 && self.z_dim.n() >= 2);
        debug_assert!(self.x_dim.max() > self.x_dim.min());
        debug_assert!(self.y_dim.max() > self.y_dim.min());
        debug_assert!(self.z_dim.max() > self.z_dim.min());
        debug_assert!(self.x_dim.delta().is_finite() && self.x_dim.delta() > 0.0);
        debug_assert!(self.y_dim.delta().is_finite() && self.y_dim.delta() > 0.0);
        debug_assert!(self.z_dim.delta().is_finite() && self.z_dim.delta() > 0.0);

        #[cfg(feature = "bounds-check")]
        {
            // internal consistency & lengths
            assert_eq!(
                self.node_volume.len(),
                self.x_dim.n() * self.y_dim.n() * self.z_dim.n(),
                "node_volume length mismatch vs dims"
            );
            // node_volume should be positive everywhere
            assert!(self.node_volume.data().iter().all(|&v| v.is_finite() && v > 0.0));
        }
    }

    #[inline(always)]
    fn check_real_coord_in_bounds(&self, real_coord: DVec3) {
        // cheap: bounds check (debug only)
        debug_assert!(
            real_coord.x >= self.x_dim.min() && real_coord.x <= self.x_dim.max() &&
            real_coord.y >= self.y_dim.min() && real_coord.y <= self.y_dim.max() &&
            real_coord.z >= self.z_dim.min() && real_coord.z <= self.z_dim.max(),
            "real_coord out of bounds: {real_coord:?}"
        );

        #[cfg(feature = "bounds-check")]
        {
            assert!(
                real_coord.x.is_finite() && real_coord.y.is_finite() && real_coord.z.is_finite(),
                "real_coord must be finite: {real_coord:?}"
            );
        }
    }

    pub fn new(x_dim: SingleDimSpec, y_dim: SingleDimSpec, z_dim: SingleDimSpec)
               -> Self {
        // Node volumes are reduced along faces, edges, corners
        let vol = x_dim.delta * y_dim.delta * z_dim.delta;
        let mut node_volume = ThreeDField::new(x_dim.n, y_dim.n, z_dim.n, vol);

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
        let spec = Self { x_dim, y_dim, z_dim, node_volume };
        spec.check_dims_basic();
        spec
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

    #[inline(always)]
    fn full_node_index_unchecked(&self, real_coord : DVec3) -> DVec3 {
        DVec3::new(
            (real_coord.x - self.x_dim.min()) / self.x_dim.delta(),
            (real_coord.y - self.y_dim.min()) / self.y_dim.delta(),
            (real_coord.z - self.z_dim.min()) / self.z_dim.delta(),
        )
    }

    pub fn get_full_node_index(&self, real_coord : DVec3) -> DVec3 {
        self.check_real_coord_in_bounds(real_coord);
        self.full_node_index_unchecked(real_coord)
    }

    pub fn get_full_node_index_no_assert(&self, real_coord : DVec3) -> DVec3 {
        self.full_node_index_unchecked(real_coord)
    }

}

impl std::fmt::Display for ThreeDWorldSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Three dimensional world mesh with dimensions:")?;
        writeln!(f, "X: {}", self.x_dim)?;
        writeln!(f, "Y: {}", self.y_dim)?;
        writeln!(f, "Z: {}", self.z_dim)?;
        Ok(())
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

    #[inline(always)]
    fn check_world_shapes(&self) {
        let nx = self.world_spec.x_dim().n();
        let ny = self.world_spec.y_dim().n();
        let nz = self.world_spec.z_dim().n();

        debug_assert_eq!(self.phi.len(), nx * ny * nz);
        debug_assert_eq!(self.rho.len(), nx * ny * nz);
        debug_assert_eq!(self.ef.len(),  nx * ny * nz);

        #[cfg(feature = "bounds-check")]
        {
            // check spec consistency too
            self.world_spec.check_dims_basic();

            // verify all dt etc are reasonable
            assert!(self.time.dt.is_finite() && self.time.dt > 0.0, "dt must be >0 and finite");
            assert!(EPS0.is_finite() && EPS0 > 0.0, "EPS0 must be >0 and finite");

            // sanity: phi/rho finite if you rely on them numerically
            assert!(self.phi.data().iter().all(|&v| v.is_finite()));
            assert!(self.rho.data().iter().all(|&v| v.is_finite()));
            assert!(self.ef.data().iter().all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite()));
        }
    }

    #[inline(always)]
    fn check_min_dims_for_stencils(&self) {
        // compute_ef uses +/-2 at boundaries => need at least 3 points in each dim
        debug_assert!(self.world_spec.x_dim().n() >= 3);
        debug_assert!(self.world_spec.y_dim().n() >= 3);
        debug_assert!(self.world_spec.z_dim().n() >= 3);

        #[cfg(feature = "bounds-check")]
        {
            assert!(self.world_spec.x_dim().n() >= 3, "x_dim.n must be >= 3 for compute_ef stencils");
            assert!(self.world_spec.y_dim().n() >= 3, "y_dim.n must be >= 3 for compute_ef stencils");
            assert!(self.world_spec.z_dim().n() >= 3, "z_dim.n must be >= 3 for compute_ef stencils");
        }
    }


    pub fn new(world_spec: ThreeDWorldSpec, dt: f64) -> Self {
        let nx = world_spec.x_dim().n();
        let ny = world_spec.y_dim().n();
        let nz = world_spec.z_dim().n();
        
        let world = Self {
            world_spec,
            time: TimeRepresentation{iteration: 0, sim_time: 0.0, wall_time: 0.0,
                                     wall_timer: Instant::now(), dt: dt},
            phi: ThreeDField::new(nx, ny, nz, 0.0),
            rho: ThreeDField::new(nx, ny, nz, 0.0),
            ef: ThreeDField::new(nx, ny, nz, DVec3::new(0.0, 0.0, 0.0)),
        };
        world.check_world_shapes();
        world.check_min_dims_for_stencils();
        world
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

    // A couple helper functions, used in testing
    pub(crate) fn phi_mut(&mut self) -> &mut ThreeDField<f64> { &mut self.phi }
    pub(crate) fn rho_mut(&mut self) -> &mut ThreeDField<f64> { &mut self.rho }


    // Requires that these fields are the same dim, etc
    pub fn compute_rho(&mut self, species : &[Species]) {
        self.check_world_shapes();
        #[cfg(feature = "bounds-check")]
        {
            for s in species {
                assert_eq!(s.number_density.len(), self.rho.len(), "species {} density shape mismatch", s.name);
                assert!(s.charge.is_finite(), "species {} has non-finite charge", s.name);
                assert!(s.number_density.data().iter().all(|&v| v.is_finite()), "species {} has non-finite density", s.name);
            }
        }
        self.rho.set_all(0.0);
        
        for s in species {
            self.rho.elementwise_inplace_add_scaled(s.charge, &s.number_density);
        }
        #[cfg(feature = "bounds-check")]
        {
            assert!(
                self.rho.data().iter().all(|&v| v.is_finite()),
                "rho contains non-finite values after compute_rho"
            );
        }

    }
    
    pub fn get_ef(&self, i: usize, j: usize, k: usize) -> DVec3 {
        self.ef.get(i,j,k)
    }

    pub fn interpolate_ef(&self, full_idx: DVec3) -> DVec3 {
        self.ef.linear_interpolate(full_idx)
    }
    
    pub fn solve_potential_gs_sor(&mut self, max_iter : usize, config: SorSolverConfig) -> anyhow::Result<usize> {
        self.check_world_shapes();
        debug_assert!(max_iter > 0);
        debug_assert!(self.world_spec.x_dim().n() >= 3 &&
                      self.world_spec.y_dim().n() >= 3 &&
                      self.world_spec.z_dim().n() >= 3);

        debug_assert!(config.omega > 0.0 && config.omega < 2.0);
        debug_assert!(config.l2_conv > 0.0);
        debug_assert!(config.check_every > 0);
        
        // precompute some commonly used values
        let inv_dx2 : f64 = 1.0 / (self.world_spec.x_dim().delta() * self.world_spec.x_dim().delta());
        let inv_dy2 : f64 = 1.0 / (self.world_spec.y_dim().delta() * self.world_spec.y_dim().delta());
        let inv_dz2 : f64 = 1.0 / (self.world_spec.z_dim().delta() * self.world_spec.z_dim().delta());

        let mut l2 : f64 = 1e12;

        let nx = self.world_spec.x_dim().n();
        let ny = self.world_spec.y_dim().n();
        let nz = self.world_spec.z_dim().n();

        let found = {
            let mut result = None;

            for iter in 0..max_iter {
                for i in 1..nx - 1 {
                    for j in 1..ny - 1 {
                        for k in 1..nz - 1 {
                            let phi_new = ( self.rho.get(i,j,k) / EPS0 +
                                            inv_dx2 * (self.phi.get(i-1,j,k) + self.phi.get(i+1,j,k)) +
                                            inv_dy2 * (self.phi.get(i,j-1,k) + self.phi.get(i,j+1,k)) +
                                            inv_dz2 * (self.phi.get(i,j,k-1) + self.phi.get(i,j,k+1)) ) /
                                (2.0 * inv_dx2 + 2.0 * inv_dy2 + 2.0 * inv_dz2);
                            let phi_update = (1.0 - config.omega) * self.phi.get(i,j,k) + config.omega * phi_new;
                            self.phi.set(i,j,k, phi_update);
                        }
                    }
                }

                // Periodic check for convergence
                if iter % config.check_every == 0 {
                    let mut sum: f64 = 0.0;
                    for i in 1..nx - 1 {
                        for j in 1..ny - 1 {
                            for k in 1..nz - 1 {
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
                    if l2 < config.l2_conv {
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
                Err(anyhow::anyhow!("GS SOR didn't converge.  L2 residual {l2:.6}"))
            }
        }
    }

    pub fn compute_ef(&mut self) -> anyhow::Result<()> {
        self.check_world_shapes();
        self.check_min_dims_for_stencils();

        #[cfg(feature="bounds-check")]
        {
            assert!(self.phi.data().iter().all(|&v| v.is_finite()), "phi contains NaNs or Infs");
        }
        
        let two_dx : f64 = 2.0 * self.world_spec.x_dim().delta();
        let two_dy : f64 = 2.0 * self.world_spec.y_dim().delta();
        let two_dz : f64 = 2.0 * self.world_spec.z_dim().delta();

        let mut ef: DVec3 = [0.0, 0.0, 0.0].into();

        let nx = self.world_spec.x_dim().n();
        let ny = self.world_spec.y_dim().n();
        let nz = self.world_spec.z_dim().n();
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // In each inner loop iteration, set each component of ef, then
                    // at the end, assign it to self.ef[i,j,k]
                    if i == 0 {
                        ef.x = - (-3.0 * self.phi.get(i,j,k)
                                   + 4.0 * self.phi.get(i+1, j, k)
                                   - self.phi.get(i+2, j, k))
                            / two_dx;
                    } else if i == nx - 1 {
                        ef.x = - (3.0 * self.phi.get(i,j,k)
                                   - 4.0 * self.phi.get(i-1, j, k)
                                   + self.phi.get(i-2, j, k))
                            / two_dx;
                    } else {
                        ef.x = -(self.phi.get(i+1,j,k) - self.phi.get(i-1,j,k))
                            / two_dx;
                    }
                    
                    if j == 0 {
                        ef.y = - (-3.0 * self.phi.get(i,j,k)
                                   + 4.0 * self.phi.get(i, j+1, k)
                                   - self.phi.get(i, j+2, k))
                            / two_dy;
                    } else if j == ny - 1 {
                        ef.y = - (3.0 * self.phi.get(i,j,k)
                                   - 4.0 * self.phi.get(i, j-1, k)
                                   + self.phi.get(i, j-2, k))
                            / two_dy;
                    } else {
                        ef.y = -(self.phi.get(i,j+1,k) - self.phi.get(i,j-1,k))
                            / two_dy;
                    }

                    if k == 0 {
                        ef.z = - (-3.0 * self.phi.get(i,j,k)
                                   + 4.0 * self.phi.get(i, j, k+1)
                                   - self.phi.get(i, j, k+2))
                            / two_dz;
                    } else if k == nz - 1 {
                        ef.z = - (3.0 * self.phi.get(i,j,k)
                                   - 4.0 * self.phi.get(i, j, k-1)
                                   + self.phi.get(i, j, k-2))
                            / two_dz;
                    } else {
                        ef.z = -(self.phi.get(i,j,k+1) - self.phi.get(i,j,k-1))
                            / two_dz;
                    }

                    self.ef.set(i,j,k, ef);
                }
            }
        }
        
        Ok(())
    }

    pub fn compute_potential_energy(&self) -> f64 {
        debug_assert_eq!(self.ef.len(),
                         self.world_spec.x_dim().n() *
                         self.world_spec.y_dim().n() *
                         self.world_spec.z_dim().n());

        #[cfg(feature="bounds-check")]
        {
            assert!(self.ef.data().iter().all(|v| v.x.is_finite() &&
                                              v.y.is_finite() &&
                                              v.z.is_finite()));
            assert!(self.world_spec.node_volume().data().iter().all(|&v| v.is_finite() && v > 0.0));
        }


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

        #[cfg(feature="bounds-check")]
        {
            let ef_flat = Self::flatten_dvec3(self.ef.data());
            assert_eq!(ef_flat.len(), 3 * npts,
                       "flattened ef must have 3*npts scalars");
        }

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

impl std::fmt::Display for ThreeDWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ThreeDWorld")?;
        writeln!(f, "grid:\n{}", self.world_spec)
    }
}


// Probably should set up different numbers of cells, deltas, and initial positions
// for the world, in both functions below
#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn efield_of_constant_phi_is_zero() -> anyhow::Result<()> {
        let x_dim = SingleDimSpec::new(21, -0.1, 0.1);
        let y_dim = SingleDimSpec::new(21, -0.1, 0.1);
        let z_dim = SingleDimSpec::new(21, -0.0, 0.2);
        let world_spec = ThreeDWorldSpec::new(x_dim, y_dim, z_dim);
        let dt: f64 = 2e-10;
        let mut world = ThreeDWorld::new(world_spec, dt);

        let const_val: f64 = 13.444;
        
        for i in 0..world.world_spec().x_dim().n() {
            for j in 0..world.world_spec().y_dim().n() {
                for k in 0..world.world_spec().z_dim().n() {
                    world.set_phi(i, j, k, const_val);
                }
            }
        }
        world.compute_ef().map_err(|e| anyhow::anyhow!(
            "EF computation failed with error {}", e))?;

        let tol = 1e-12;
        for i in 0..world.world_spec().x_dim().n() {
            for j in 0..world.world_spec().y_dim().n() {
                for k in 0..world.world_spec().z_dim().n() {
                    let e = world.get_ef(i,j,k);
                    assert!(e.x.abs() < tol);
                    assert!(e.y.abs() < tol);
                    assert!(e.z.abs() < tol);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn efield_of_linear_phi_is_constant() -> anyhow::Result<()> {
        let x_dim = SingleDimSpec::new(21, -0.1, 0.1);
        let y_dim = SingleDimSpec::new(21, -0.1, 0.1);
        let z_dim = SingleDimSpec::new(21, -0.0, 0.2);
        let world_spec = ThreeDWorldSpec::new(x_dim, y_dim, z_dim);
        let dt: f64 = 2e-10;
        let mut world = ThreeDWorld::new(world_spec, dt);
        
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
        world.compute_ef().map_err(|e| anyhow::anyhow!(
            "EF computation failed with error {}", e))?;

        let tol = 1e-8;
        for i in 0..world.world_spec().x_dim().n() {
            for j in 0..world.world_spec().y_dim().n() {
                for k in 0..world.world_spec().z_dim().n() {
                    let e = world.get_ef(i,j,k);
                    assert!((e.x + a).abs() < tol, "values {} and {} should only differ in sign", e.x, a );
                    assert!((e.y + b).abs() < tol, "values {} and {} should only differ in sign", e.y, b);
                    assert!((e.z + c).abs() < tol, "values {} and {} should only differ in sign", e.z, c );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn single_dim_spec_computes_delta_and_center() -> anyhow::Result<()> {
        let s = SingleDimSpec::new(11, -1.0, 1.0);
        assert_eq!(s.n(), 11);
        assert!((s.delta() - 0.2).abs() < 1e-12);
        assert!((s.center() - 0.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn world_spec_node_volume_scales_on_boundaries() -> anyhow::Result<()> {
        let x = SingleDimSpec::new(4, 0.0, 3.0);
        let y = SingleDimSpec::new(4, 0.0, 3.0);
        let z = SingleDimSpec::new(4, 0.0, 3.0);
        let ws = ThreeDWorldSpec::new(x, y, z);

        let vol = x.delta() * y.delta() * z.delta();
        let nv = ws.node_volume();

        // interior
        assert!((nv.get(1, 1, 1) - vol).abs() < 1e-12);

        // face (one boundary)
        assert!((nv.get(0, 1, 1) - vol * 0.5).abs() < 1e-12);

        // edge (two boundaries)
        assert!((nv.get(0, 0, 1) - vol * 0.25).abs() < 1e-12);

        // corner (three boundaries)
        assert!((nv.get(0, 0, 0) - vol * 0.125).abs() < 1e-12);

        Ok(())
    }

    #[test]
    fn full_node_index_maps_corners() -> anyhow::Result<()> {
        let x = SingleDimSpec::new(11, -1.0, 1.0);
        let y = SingleDimSpec::new(11, -2.0, 2.0);
        let z = SingleDimSpec::new(11,  0.0, 1.0);
        let ws = ThreeDWorldSpec::new(x, y, z);

        let min = ws.get_min_corner();
        let max = ws.get_max_corner();

        let idx_min = ws.get_full_node_index(min);
        assert!((idx_min.x - 0.0).abs() < 1e-12);
        assert!((idx_min.y - 0.0).abs() < 1e-12);
        assert!((idx_min.z - 0.0).abs() < 1e-12);

        let idx_max = ws.get_full_node_index(max);
        assert!((idx_max.x - (ws.x_dim().n() as f64 - 1.0)).abs() < 1e-12);
        assert!((idx_max.y - (ws.y_dim().n() as f64 - 1.0)).abs() < 1e-12);
        assert!((idx_max.z - (ws.z_dim().n() as f64 - 1.0)).abs() < 1e-12);

        Ok(())
    }

    #[test]
    #[should_panic]
    fn full_node_index_panics_out_of_bounds() {
        let x = SingleDimSpec::new(11, 0.0, 1.0);
        let y = SingleDimSpec::new(11, 0.0, 1.0);
        let z = SingleDimSpec::new(11, 0.0, 1.0);
        let ws = ThreeDWorldSpec::new(x, y, z);

        let _ = ws.get_full_node_index(DVec3::new(-0.1, 0.5, 0.5));
    }

    #[test]
    fn gs_sor_converges_for_zero_rho() -> anyhow::Result<()> {
        let x_dim = SingleDimSpec::new(11, -0.1, 0.1);
        let y_dim = SingleDimSpec::new(11, -0.1, 0.1);
        let z_dim = SingleDimSpec::new(11, -0.1, 0.1);
        let world_spec = ThreeDWorldSpec::new(x_dim, y_dim, z_dim);
        let mut world = ThreeDWorld::new(world_spec, 1e-9);

        // rho already zero; phi already zero
        let iters = world.solve_potential_gs_sor(2000, SorSolverConfig::default())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // It might converge at iter=0 or very quickly; just assert it did converge.
        assert!(iters < 2000);

        // Spot check phi remains ~0
        let tol = 1e-12;
        for i in 0..world.world_spec().x_dim().n() {
            for j in 0..world.world_spec().y_dim().n() {
                for k in 0..world.world_spec().z_dim().n() {
                    assert!(world.phi().get(i,j,k).abs() < tol);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn potential_energy_zero_for_zero_field() -> anyhow::Result<()> {
        let x_dim = SingleDimSpec::new(7, 0.0, 1.0);
        let y_dim = SingleDimSpec::new(7, 0.0, 1.0);
        let z_dim = SingleDimSpec::new(7, 0.0, 1.0);
        let world_spec = ThreeDWorldSpec::new(x_dim, y_dim, z_dim);
        let mut world = ThreeDWorld::new(world_spec, 1e-9);
        
        // ef initialized to zero
        let pe = world.compute_potential_energy();
        assert!(pe.abs() < 1e-18);
        Ok(())
    }

}

#[cfg(test)]
mod rho_tests {
    use super::*;
    use crate::particles::{Species, Particle};

    fn make_world(nx: usize, ny: usize, nz: usize) -> ThreeDWorld {
        let x = SingleDimSpec::new(nx, 0.0, 1.0);
        let y = SingleDimSpec::new(ny, 0.0, 1.0);
        let z = SingleDimSpec::new(nz, 0.0, 1.0);
        let spec = ThreeDWorldSpec::new(x, y, z);
        ThreeDWorld::new(spec, 1e-9)
    }

    #[test]
    fn compute_rho_empty_species_is_zero() {
        let mut world = make_world(5, 4, 3);

        world.compute_rho(&[]);

        for i in 0..world.world_spec().x_dim().n() {
            for j in 0..world.world_spec().y_dim().n() {
                for k in 0..world.world_spec().z_dim().n() {
                    assert_eq!(world.rho().get(i, j, k), 0.0);
                }
            }
        }
    }

    #[test]
    fn compute_rho_single_species_matches_charge_times_density() {
        let (nx, ny, nz) = (5, 4, 3);
        let mut world = make_world(nx, ny, nz);

        let x = SingleDimSpec::new(nx, 0.0, 1.0);
        let y = SingleDimSpec::new(ny, 0.0, 1.0);
        let z = SingleDimSpec::new(nz, 0.0, 1.0);
        let mut s = Species::new("ions", 2.0, 2.0, x, y, z);

        // Make density vary so we’re not just testing a constant field
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let nd = (i as f64) + 10.0 * (j as f64) + 100.0 * (k as f64);
                    s.number_density.set(i, j, k, nd);
                }
            }
        }

        world.compute_rho(&vec![s]);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let nd = (i as f64) + 10.0 * (j as f64) + 100.0 * (k as f64);
                    let expected = 2.0 * nd;
                    let got = world.rho().get(i, j, k);
                    assert!((got - expected).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn compute_rho_two_species_sums_correctly_with_negative_charge() {
        let (nx, ny, nz) = (5, 4, 3);
        let mut world = make_world(nx, ny, nz);

        let x = SingleDimSpec::new(nx, 0.0, 1.0); // These will copy into below
        let y = SingleDimSpec::new(ny, 0.0, 1.0);
        let z = SingleDimSpec::new(nz, 0.0, 1.0);
        let mut ions = Species::new("ions", 1.0, 1.0, x, y, z);
        let mut elec = Species::new("electrons", 0.1, -1.0, x, y, z);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let n1 = 1.0 + (i as f64);
                    let n2 = 0.5 + 2.0 * (j as f64);
                    ions.number_density.set(i, j, k, n1);
                    elec.number_density.set(i, j, k, n2);
                }
            }
        }

        world.compute_rho(&vec![ions, elec]);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let n1 = 1.0 + (i as f64);
                    let n2 = 0.5 + 2.0 * (j as f64);
                    let expected = 1.0 * n1 + (-1.0) * n2;
                    let got = world.rho().get(i, j, k);
                    assert!((got - expected).abs() < 1e-12);
                }
            }
        }
    }

    // Optional: only if your ThreeDField elementwise ops check shape and panic on mismatch.
    #[test]
    #[should_panic]
    fn compute_rho_panics_on_density_shape_mismatch() {
        let mut world = make_world(5, 4, 3);

        let (nx, ny, nz) = (6, 4, 3);

        let x = SingleDimSpec::new(nx, 0.0, 1.0);
        let y = SingleDimSpec::new(ny, 0.0, 1.0);
        let z = SingleDimSpec::new(nz, 0.0, 1.0);

        // Deliberately wrong dimensions
        let bad = Species::new("bad", 1.0, 1.0, x, y, z);

        world.compute_rho(&vec![bad]);
    }
    
    #[test]
    fn compute_rho_resets_previous_values() {
        let (nx, ny, nz) = (5, 4, 3);
        let mut world = make_world(nx, ny, nz);
        
        // poison rho with junk
        world.rho_mut().set_all(123.0);
        
        let x = SingleDimSpec::new(nx, 0.0, 1.0);
        let y = SingleDimSpec::new(ny, 0.0, 1.0);
        let z = SingleDimSpec::new(nz, 0.0, 1.0);
        let mut s = Species::new("ions", 1.0, 2.0, x, y, z);
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    s.number_density.set(i, j, k, 1.0);
                }
            }
        }
        
        world.compute_rho(&[s]);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    assert!((world.rho().get(i,j,k) - 2.0).abs() < 1e-12);
                }
            }
        }
    }

}

#[cfg(test)]
mod poisson_tests {
    use super::*;
    use glam::DVec3;

    fn make_world(n: usize) -> ThreeDWorld {
        let x = SingleDimSpec::new(n, 0.0, 1.0);
        let y = SingleDimSpec::new(n, 0.0, 1.0);
        let z = SingleDimSpec::new(n, 0.0, 1.0);
        let spec = ThreeDWorldSpec::new(x, y, z);
        ThreeDWorld::new(spec, 1e-9)
    }

    #[test]
    fn poisson_zero_rho_stays_zero_and_converges() -> anyhow::Result<()> {
        let mut world = make_world(11);

        // Ensure phi and rho are zero
        world.phi_mut().set_all(0.0);
        world.rho_mut().set_all(0.0);

        let iters = world
            .solve_potential_gs_sor(5000, SorSolverConfig::default())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        assert!(iters < 5000);

        let tol = 1e-12;
        let nx = world.world_spec().x_dim().n();
        let ny = world.world_spec().y_dim().n();
        let nz = world.world_spec().z_dim().n();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    assert!(world.phi().get(i, j, k).abs() < tol);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn poisson_manufactured_solution_recovers_phi() -> anyhow::Result<()> {
        let n = 17; // a bit larger gives cleaner interior
        let mut world = make_world(n);

        let dx = world.world_spec().x_dim().delta();
        let dy = world.world_spec().y_dim().delta();
        let dz = world.world_spec().z_dim().delta();
        let inv_dx2 = 1.0 / (dx * dx);
        let inv_dy2 = 1.0 / (dy * dy);
        let inv_dz2 = 1.0 / (dz * dz);

        // True phi: sin(pi x) sin(pi y) sin(pi z); boundaries are 0 automatically
        let a = 1.23;
        let nx = world.world_spec().x_dim().n();
        let ny = world.world_spec().y_dim().n();
        let nz = world.world_spec().z_dim().n();

        let x0 = world.world_spec().x_dim().min();
        let y0 = world.world_spec().y_dim().min();
        let z0 = world.world_spec().z_dim().min();

        // Fill phi_true into world.phi, but we'll later reset phi to 0 as initial guess.
        let mut phi_true = ThreeDField::new(nx, ny, nz, 0.0);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = x0 + (i as f64) * dx;
                    let y = y0 + (j as f64) * dy;
                    let z = z0 + (k as f64) * dz;
                    let val = a
                        * (std::f64::consts::PI * x).sin()
                        * (std::f64::consts::PI * y).sin()
                        * (std::f64::consts::PI * z).sin();
                    phi_true.set(i, j, k, val);
                }
            }
        }

        // Compute rho on interior from discrete Laplacian that matches your residual
        world.rho_mut().set_all(0.0);
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let phi_c = phi_true.get(i, j, k);
                    let lap_phi = phi_c * (2.0 * inv_dx2 + 2.0 * inv_dy2 + 2.0 * inv_dz2)
                        - inv_dx2 * (phi_true.get(i - 1, j, k) + phi_true.get(i + 1, j, k))
                        - inv_dy2 * (phi_true.get(i, j - 1, k) + phi_true.get(i, j + 1, k))
                        - inv_dz2 * (phi_true.get(i, j, k - 1) + phi_true.get(i, j, k + 1));

                    let rho = EPS0 * lap_phi;
                    world.rho_mut().set(i, j, k, rho);
                }
            }
        }

        // Boundary phi is 0. Start from phi=0 everywhere.
        world.phi_mut().set_all(0.0);

        let iters = world
            .solve_potential_gs_sor(20000, SorSolverConfig::default())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        assert!(iters < 20000);

        // Compare interior
        let tol = 5e-6; // SOR should reach ~1e-6 L2; pointwise may be a bit looser
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let got = world.phi().get(i, j, k);
                    let exp = phi_true.get(i, j, k);
                    assert!(
                        (got - exp).abs() < tol,
                        "mismatch at ({i},{j},{k}): got={got}, exp={exp}"
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn poisson_returns_err_when_max_iter_too_small() -> anyhow::Result<()> {
        let mut world = make_world(11);

        // Put a simple localized charge in the interior
        world.rho_mut().set_all(0.0);
        world.phi_mut().set_all(0.0);
        world.rho_mut().set(5, 5, 5, 1e-6);

        let res = world.solve_potential_gs_sor(1, SorSolverConfig::default()); // intentionally too small
        assert!(res.is_err());

        Ok(())
    }

    #[test]
    fn poisson_does_not_modify_boundary_phi() -> anyhow::Result<()> {
        let mut world = make_world(11);
        let nx = world.world_spec().x_dim().n();
        let ny = world.world_spec().y_dim().n();
        let nz = world.world_spec().z_dim().n();
        
        world.rho_mut().set_all(0.0);
        world.phi_mut().set_all(0.0);
        
        // Mark each face interior with a unique value (avoid overlaps)
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                world.phi_mut().set(i, j, 0,      1.0);
                world.phi_mut().set(i, j, nz - 1, 2.0);
            }
        }
        for i in 1..nx-1 {
            for k in 1..nz-1 {
                world.phi_mut().set(i, 0,      k, 3.0);
                world.phi_mut().set(i, ny - 1, k, 4.0);
            }
        }
        for j in 1..ny-1 {
            for k in 1..nz-1 {
                world.phi_mut().set(0,      j, k, 5.0);
                world.phi_mut().set(nx - 1, j, k, 6.0);
            }
        }

        let _iters = world.solve_potential_gs_sor(5000, SorSolverConfig::default())
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Verify face interiors unchanged
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                assert_eq!(world.phi().get(i, j, 0),      1.0);
                assert_eq!(world.phi().get(i, j, nz - 1), 2.0);
            }
        }
        for i in 1..nx-1 {
            for k in 1..nz-1 {
                assert_eq!(world.phi().get(i, 0,      k), 3.0);
                assert_eq!(world.phi().get(i, ny - 1, k), 4.0);
            }
        }
        for j in 1..ny-1 {
            for k in 1..nz-1 {
                assert_eq!(world.phi().get(0,      j, k), 5.0);
                assert_eq!(world.phi().get(nx - 1, j, k), 6.0);
            }
        }
        
        Ok(())
    }

}
