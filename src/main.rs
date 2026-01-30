use anyhow::Result;
use std::path::PathBuf;

use kinetic_pic_electrostatic_3d::world_3d::{ThreeDWorldSpec, SingleDimSpec};


fn set_phi_to_test_values(world : &mut ThreeDWorldSpec) {
    for j in 0..world.get_y_dim_n() {
        for k in 0..world.get_z_dim_n() {
            world.set_phi(0,j,k, 1.0);
        }
    }

    for i in 0..world.get_x_dim_n() {
        for j in 0..world.get_y_dim_n() {
            world.set_phi(i,j,0, 2.0);
        }
    }
}


fn main() -> Result <()> {
    let x_dim = match SingleDimSpec::init(21, -0.1, 0.1) {
        Ok(s) => s,
        Err(_) => {return Err(anyhow::anyhow!("bad 3d spec"));}
    };

    let y_dim = match SingleDimSpec::init(21, -0.1, 0.1) {
        Ok(s) => s,
        Err(_) => {return Err(anyhow::anyhow!("bad 3d spec"));}
    };

    let z_dim = match SingleDimSpec::init(21, -0.0, 0.2) {
        Ok(s) => s,
        Err(_) => {return Err(anyhow::anyhow!("bad 3d spec"));}
    };

    let mut world = match ThreeDWorldSpec::init(x_dim, y_dim, z_dim) {
        Ok(s) => s,
        Err(_) => {
            println!("Failed to create a three d world spec");
            return Err(anyhow::anyhow!("bad 3d spec"));
        }
    };

    set_phi_to_test_values(&mut world);

    world.solve_potential_gs_sor(5000);
    world.compute_ef();
    
    world.print()?;

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = root.join("images").join("world_fields.vti");
    world.write_world_vti(out_dir)?;
    
    Ok(())
}
