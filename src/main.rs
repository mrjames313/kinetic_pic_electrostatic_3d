use anyhow::Result;

use kinetic_pic_electrostatic_3d::world_3d::{ThreeDWorldSpec, SingleDimSpec};

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

    let world = match ThreeDWorldSpec::init(x_dim, y_dim, z_dim) {
        Ok(s) => s,
        Err(_) => {
            println!("Failed to create a three d world spec");
            return Err(anyhow::anyhow!("bad 3d spec"));
        }
    };

    world.print()?;
    Ok(())
}
