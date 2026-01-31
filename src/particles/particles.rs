use anyhow::Result;
use glam::DVec3;

use crate::world_3d::ThreeDField;
use crate::world_3d::SingleDimSpec;


pub struct Particle {
    pos: DVec3,
    vel: DVec3,
    macroparticle_weight: f64,
}


pub struct Species {
    name: String,  // Look into OnceCell<String> to enforce types
    mass: f64,
    charge: f64,
    number_density: ThreeDField<f64>,
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
}
    
                    
    
    

