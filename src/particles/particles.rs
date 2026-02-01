use anyhow::Result;
use glam::DVec3;
use rand::Rng;

use crate::world_3d::ThreeDField;
use crate::world_3d::SingleDimSpec;
use crate::world_3d::ThreeDWorldSpec;

pub struct Particle {
    pos: DVec3,
    vel: DVec3,
    macroparticle_weight: f64,
}


pub struct Species {
    name: String,  // Look into OnceCell<String> to enforce types
    mass: f64,
    pub charge: f64,
    pub number_density: ThreeDField<f64>,
    particles: Vec<Particle>,
}

impl Species {
    pub fn init(name: String, mass: f64, charge: f64,
                x_dim: SingleDimSpec, y_dim: SingleDimSpec, z_dim: SingleDimSpec) -> Result<Self> {
        let num_den = ThreeDField::init(x_dim.n, y_dim.n, z_dim.n, 0.0);
        let p = Vec::new();
        let s = Self{name:name, mass:mass, charge:charge,
                     number_density:num_den, particles:p };
        Ok(s)
    }

    pub fn get_num_particles(&self) -> usize {
        self.particles.len()
    }
    
    // Helper function to load a set of particles according to the density, num particles,
    // and specified box.
    pub fn load_particles_box(&mut self, corner_min: DVec3, corner_max: DVec3,
                              number_density: f64, num_sim_particles: usize) -> Result <()> {
        anyhow::ensure!(corner_min.x < corner_max.x,
                        "x dim of corner_max must be greater than corner_min)");
        anyhow::ensure!(corner_min.y < corner_max.y,
                        "y dim of corner_max must be greater than corner_min)");
        anyhow::ensure!(corner_min.z < corner_max.z,
                        "z dim of corner_max must be greater than corner_min)");

        let x_extent = corner_max.x - corner_min.x;
        let y_extent = corner_max.y - corner_min.z;
        let z_extent = corner_max.y - corner_min.z;
        
        let box_vol = x_extent * y_extent * z_extent;
        let num_actual_particles = number_density * box_vol; // leave it as f64
        let macroparticle_weight = num_actual_particles / num_sim_particles as f64; // this is different than book - pg74
        
        self.particles.reserve(num_sim_particles);

        let mut rng = rand::thread_rng();
        let mut pos: DVec3 = [0.0, 0.0, 0.0].into();
        for i in 0..num_sim_particles {
            pos.x = corner_min.x + rng.gen_range(0.0 .. 1.0) * x_extent;
            pos.x = corner_min.y + rng.gen_range(0.0 .. 1.0) * y_extent;
            pos.x = corner_min.z + rng.gen_range(0.0 .. 1.0) * z_extent;
            self.particles.push(Particle{pos:pos, vel:[0.0, 0.0, 0.0].into(),
                                         macroparticle_weight:macroparticle_weight});
        }
        Ok(())
    }

    // TODO: Consider if passing world in here is the best design
    pub fn compute_number_density(&mut self, world : &ThreeDWorldSpec) {
        self.number_density.set_all(0.0);
        for particle in self.particles.iter() {
            let full_idx : DVec3 = world.get_full_node_index(particle.pos);
            self.number_density.distribute(full_idx, particle.macroparticle_weight);
        }
        // TODO: think about whether divide is the right operation here
        self.number_density.elementwise_inplace_div(&world.node_volume);
    }
}
