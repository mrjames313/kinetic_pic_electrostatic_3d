use anyhow::Result;
use std::path::PathBuf;

use kinetic_pic_electrostatic_3d::constants::*;
use kinetic_pic_electrostatic_3d::world_3d::{ThreeDWorldSpec, SingleDimSpec};
use kinetic_pic_electrostatic_3d::particles::Species;
use kinetic_pic_electrostatic_3d::output::{CsvLogger, DiagnosticOutput, WriteVti};


fn main() -> Result <()> {

    let dt = 2.0e-10;
    
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

    let mut world = match ThreeDWorldSpec::init(x_dim, y_dim, z_dim, dt) {
        Ok(s) => s,
        Err(_) => {
            println!("Failed to create a three d world spec");
            return Err(anyhow::anyhow!("bad 3d spec"));
        }
    };

    x_dim.print();
    y_dim.print();
    z_dim.print();

    println!("World");
    world.print_spec();

    // don't know if this is necessary before starting sim, but doesn't hurt
    world.solve_potential_gs_sor(5000);
    world.compute_ef();
    
    world.print()?;

    // now introduce particles to the system
    let mut ions = match Species::init("O+".to_string(), 16.0 * AMU, QE, x_dim, y_dim, z_dim) {
        Ok(s) => s,
        Err(_) => {
            println!("Failed to create ions species");
            return Err(anyhow::anyhow!("Bad ions species spec"));
        }
    };
    
    let mut electrons = match Species::init("e".to_string(), ME, -1.0 * QE, x_dim, y_dim, z_dim) {
        Ok(s) => s,
        Err(_) => {
            println!("Failed to create electrons species");
            return Err(anyhow::anyhow!("Bad electrons species spec"));
        }
    };


    let np_ions: usize = 80_000;
    let np_electrons: usize = 10_000;
    let num_den: f64 = 1.0e11;

    ions.load_particles_box(world.get_min_corner(), world.get_max_corner(),
                            num_den, np_ions, &world);

    electrons.load_particles_box(world.get_min_corner(), world.get_center(),
                                 num_den, np_electrons, &world);

    println!("Now have {} ions, and {} electrons loaded",
             ions.get_num_particles(), electrons.get_num_particles());

    let mut all_species : Vec<Species> = [ions, electrons].into();

    // now compute the fields with these particles in place
    for s in all_species.iter_mut() {
        s.compute_number_density(&world);
    }
    
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = root.join("images").join("world_fields.vti");
    world.write_world_vti(out_dir)?;

//    let diag_output = DiagnosticOutput;
    
    let num_iterations = 10_000;
    let write_vti = WriteVti;
    let species_dir = root.join("images").join("species_sequence");
    let species_prefix = "species_fields";

    let mut logger = CsvLogger::new(root.join("logs"))?;
                           
    world.start_iteration_time();
    while world.get_iteration() < num_iterations {
        // TODO: Where does this belong
        world.advance_iteration();

        world.compute_rho(&mut all_species);
    
        world.solve_potential_gs_sor(5000);
        world.compute_ef();
        for s in all_species.iter_mut() {
            s.advance(&world);
            s.compute_number_density(&world);
        }
        
        logger.log(&world, &all_species);
        let iter = world.get_iteration(); // TODO: feels a bit sloppy
        if iter % 10 == 0 {
            println!("Iter {iter}");
            write_vti.write_species_at_time_to_vti(&world, &all_species, iter, &species_dir, &species_prefix);
            //diag_output.print_status(&world, &all_species);
        }
    }
    
    Ok(())
}
