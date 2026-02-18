use anyhow::Result;
use std::path::PathBuf;

use kinetic_pic_electrostatic_3d::world_3d::{ThreeDWorld, ThreeDWorldSpec, SingleDimSpec, SorSolverConfig};


// sets up phi for two of the cube sides to non-zero.  Remaining four
// sides are zero
fn set_phi_to_test_values(world : &mut ThreeDWorld) {
    for j in 0..world.world_spec().y_dim().n() {
        for k in 0..world.world_spec().z_dim().n() {
            world.set_phi(0,j,k, 1.0);
        }
    }

    for i in 0..world.world_spec().x_dim().n() {
        for j in 0..world.world_spec().y_dim().n() {
            world.set_phi(i,j,0, 2.0);
        }
    }
}


fn main() -> Result <()> {

    let x_dim = SingleDimSpec::new(21, -0.1, 0.1);
    let y_dim = SingleDimSpec::new(21, -0.1, 0.1);
    let z_dim = SingleDimSpec::new(21, -0.0, 0.2);
    let world_spec = ThreeDWorldSpec::new(x_dim, y_dim, z_dim);
    let dt: f64 = 2e-10;
    let mut world = ThreeDWorld::new(world_spec, dt);

    println!("X: {}", x_dim);
    println!("Y: {}", y_dim);
    println!("Z: {}", z_dim);

    println!("World: {}", world);
    
    set_phi_to_test_values(&mut world);

    world.solve_potential_gs_sor(5000, SorSolverConfig::default()).map_err(anyhow::Error::msg)?;
    world.compute_ef().map_err(anyhow::Error::msg)?;

    println!("World: {}", world);

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = root.join("images").join("world_fields.vti");
    world.write_world_vti(out_dir)?;
    
    Ok(())
}
