use anyhow::Result;
use glam::DVec3;
use rand::Rng;

// temp
use std::collections::HashMap;

use crate::world_3d::ThreeDField;
use crate::world_3d::SingleDimSpec;
use crate::world_3d::ThreeDWorldSpec;
use crate::output::SpeciesInfo;

pub struct Particle {
    pos: DVec3,
    vel: DVec3,
    macroparticle_weight: f64,
}


pub struct Species {
    pub name: String,  // Look into OnceCell<String> to enforce types
    mass: f64,
    pub charge: f64,
    pub number_density: ThreeDField<f64>,
    particles: Vec<Particle>,
}


pub fn get_species_info_from_species<'a>(sp: &'a Species) -> SpeciesInfo<'a> {
    let mp_count = sp.get_num_particles();
    let real_count = sp.get_real_count();
    let total_momentum = sp.get_momentum();
    let kinetic_e = sp.get_kinetic_energy();
    SpeciesInfo {
        name: sp.name.as_str(),
        mp_count,
        real_count,
        momentum_x: total_momentum.x,
        momentum_y: total_momentum.y,
        momentum_z: total_momentum.z,
        kinetic_e,
    }
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

    // TODO: Figure out if this will ever change - are we going to lose weight / mass?
    pub fn get_real_count(&self) -> f64 {
        let sum: f64 = self.particles.iter().map(|p| p.macroparticle_weight).sum();
        sum
    }

    // TODO: understand this: total_momentum: DVec3 = sp.particles.iter().fold(DVec3::ZERO, |acc, p| {
//        acc + (sp.mass * p.macroparticle_weight) * p.vel
    pub fn get_momentum(&self) -> DVec3 {
        let mut mom: DVec3 = [0.0, 0.0, 0.0].into();
        for p in self.particles.iter() {
            mom += p.macroparticle_weight * p.vel;
        }
        mom
    }
// TODO: understand this: kinetic_e: f64 = sp.particles.iter().map(|p| {
//        0.5 * sp.mass * p.macroparticle_weight * p.vel.length_squared()
//    }).sum();
    pub fn get_kinetic_energy(&self) -> f64 {
        // 1/2 m v^2
        let mut ke: f64 = 0.0;
        for p in self.particles.iter() {
            ke += p.vel.length_squared() * p.macroparticle_weight;
        }
        ke *= 0.5 * self.mass;
        ke
    }

    
    // Helper function to load a set of particles according to the density, num particles,
    // and specified box.
    pub fn load_particles_box(&mut self, corner_min: DVec3, corner_max: DVec3,
                              number_density: f64, num_sim_particles: usize,
                              world: &ThreeDWorldSpec) -> Result <()> {
        anyhow::ensure!(corner_min.x < corner_max.x,
                        "x dim of corner_max must be greater than corner_min)");
        anyhow::ensure!(corner_min.y < corner_max.y,
                        "y dim of corner_max must be greater than corner_min)");
        anyhow::ensure!(corner_min.z < corner_max.z,
                        "z dim of corner_max must be greater than corner_min)");

        println!("Loading box with corners [{}, {}, {}] to [{}, {}, {}]",
                 corner_min.x, corner_min.y, corner_min.z,
                 corner_max.x, corner_max.y, corner_max.z);
        
        let x_extent = corner_max.x - corner_min.x;
        let y_extent = corner_max.y - corner_min.y;
        let z_extent = corner_max.z - corner_min.z;
        
        let box_vol = x_extent * y_extent * z_extent;
        let num_actual_particles = number_density * box_vol; // leave it as f64
        let macroparticle_weight = num_actual_particles / num_sim_particles as f64; // this is different than book - pg74
        
        self.particles.reserve(num_sim_particles);

        let mut rng = rand::thread_rng();
        let mut pos: DVec3 = [0.0, 0.0, 0.0].into();
        let mut vel: DVec3 = [0.0, 0.0, 0.0].into(); // Do we always assume vel starts at 0 before rewind?
        // precompute some factors for velocity rewind
        let vel_rewind_factor = 0.5 * world.get_dt() * self.charge / self.mass;
        
        for _i in 0..num_sim_particles {
            pos.x = corner_min.x + rng.gen_range(0.0 .. 1.0) * x_extent;
            pos.y = corner_min.y + rng.gen_range(0.0 .. 1.0) * y_extent;
            pos.z = corner_min.z + rng.gen_range(0.0 .. 1.0) * z_extent;

            let full_idx : DVec3 = world.get_full_node_index(pos);
            let ef = world.interpolate_ef(full_idx);
            vel -= vel_rewind_factor * ef;

            self.particles.push(Particle{pos:pos, vel:vel,
                                         macroparticle_weight:macroparticle_weight});
        }
        Ok(())
    }

    // TODO: Consider if passing world in here is the best design
    pub fn compute_number_density(&mut self, world : &ThreeDWorldSpec) {
        self.number_density.set_all(0.0);
        //        let mut rng = rand::thread_rng(); // for getting a random sample

        // debugging
        let mut counts: HashMap<[usize; 3], usize> = HashMap::new();
        
//        println!("Computing density for {} particles", self.particles.len());
        for particle in self.particles.iter() {
//            if rng.gen_range(0.0 .. 1.0) < 0.001 {
//                println!("Particle at pos [{}, {}, {}] with weight {}", particle.pos[0],
//                         particle.pos[1], particle.pos[2], particle.macroparticle_weight);
//            }
            
            let full_idx : DVec3 = world.get_full_node_index(particle.pos);

            // debug
            let key = [full_idx[0] as usize, full_idx[1] as usize, full_idx[2] as usize];
            *counts.entry(key).or_insert(0) += 1;

//            // TODO: figure out if we need to put testing around this
//            if (full_idx[0] as usize == 5 && full_idx[1] as usize == 5 && full_idx[2] as usize == 5) {
//                println!("At 5,5,5, distributing weight {}", particle.macroparticle_weight);
//            }
            
            self.number_density.distribute(full_idx, particle.macroparticle_weight);
        }
        // TODO: think about whether divide is the right operation here
        self.number_density.elementwise_inplace_div(&world.node_volume);

// TOOD: Figure out if testing is needed for this debug code        
//        // get some stats
//        let min = counts.values().min().copied().unwrap();
//        let max = counts.values().max().copied().unwrap();
//        let sum = counts.values().sum().copied();
//        println!("Stats of box counts: min {}, max {}", min, max);
    }

    pub fn advance(&mut self, world : &ThreeDWorldSpec) {
        let dt = world.get_dt();
        let charge_per_mass = self.charge / self.mass;
        
        for particle in self.particles.iter_mut() {
            let mut full_idx : DVec3 = world.get_full_node_index(particle.pos);
            let ef = world.interpolate_ef(full_idx);
            particle.vel += ef * (dt * charge_per_mass);
            particle.pos += particle.vel * dt;

            // reflect particles that go out of domain, on each of the axes
            // have to do this one-by-one due to how I structured the spec
            
            // before reflection, the index may be out-of-bounds, so call no_assert
            full_idx = world.get_full_node_index_no_assert(particle.pos);

            if full_idx[0] < 0.0 {
                particle.pos[0] = 2.0 * world.x_dim.min - particle.pos[0];
                particle.vel[0] *= -1.0;
            } else if full_idx[0] >= (world.x_dim.n-1) as f64 {
                particle.pos[0] = 2.0 * world.x_dim.max - particle.pos[0];
                particle.vel[0] *= -1.0;
            }
            if full_idx[1] < 0.0 {
                particle.pos[1] = 2.0 * world.y_dim.min - particle.pos[1];
                particle.vel[1] *= -1.0;
            } else if full_idx[1] >= (world.y_dim.n-1) as f64 {
                particle.pos[1] = 2.0 * world.y_dim.max - particle.pos[1];
                particle.vel[1] *= -1.0;
            }
            if full_idx[2] < 0.0 {
                particle.pos[2] = 2.0 * world.z_dim.min - particle.pos[2];
                particle.vel[2] *= -1.0;
            } else if full_idx[2] >= (world.z_dim.n-1) as f64 {
                particle.pos[2] = 2.0 * world.z_dim.max - particle.pos[2];
                particle.vel[2] *= -1.0;
            }
        }
    }
}
