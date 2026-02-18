use anyhow::Result;
use std::path::PathBuf;

use kinetic_pic_electrostatic_3d::constants::*;
use kinetic_pic_electrostatic_3d::world_3d::{ThreeDWorld, ThreeDWorldSpec, SingleDimSpec, SorSolverConfig};
use kinetic_pic_electrostatic_3d::particles::Species;
use kinetic_pic_electrostatic_3d::output::{CsvLogger, DiagnosticOutput, WriteVti};


fn main() -> Result <()> {

    let dt = 2.0e-10;
    
    let x_dim = SingleDimSpec::new(21, -0.1, 0.1);
    let y_dim = SingleDimSpec::new(21, -0.1, 0.1);
    let z_dim = SingleDimSpec::new(21, -0.0, 0.2);
    let world_spec = ThreeDWorldSpec::new(x_dim, y_dim, z_dim);
    let mut world = ThreeDWorld::new(world_spec, dt);

    println!("X: {}", x_dim);
    println!("Y: {}", y_dim);
    println!("Z: {}", z_dim);

    println!("World: {}", world);

    // don't know if this is necessary before starting sim, but doesn't hurt
    world.solve_potential_gs_sor(5000, SorSolverConfig::default()).map_err(anyhow::Error::msg)?;
    world.compute_ef().map_err(anyhow::Error::msg)?;

    println!("World: {}", world);

    // now introduce particles to the system
    let mut ions = Species::new("O+".to_string(), 16.0 * AMU, QE, x_dim, y_dim, z_dim);    
    let mut electrons = Species::new("e".to_string(), ME, -1.0 * QE, x_dim, y_dim, z_dim);

    let np_ions: usize = 80_000;
    let np_electrons: usize = 10_000;
    let num_den: f64 = 1.0e11;

    ions.load_particles_box(world.world_spec().get_min_corner(), world.world_spec().get_max_corner(),
                            num_den, np_ions, &world)?;

    electrons.load_particles_box(world.world_spec().get_min_corner(), world.world_spec().get_center(),
                                 num_den, np_electrons, &world)?;

    println!("Now have {} ions, and {} electrons loaded",
             ions.get_num_particles(), electrons.get_num_particles());

    let mut all_species : Vec<Species> = [ions, electrons].into();

    // now compute the fields with these particles in place
    for s in all_species.iter_mut() {
        s.compute_number_density(&world.world_spec());
    }
    
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = root.join("images").join("world_fields.vti");
    world.write_world_vti(out_dir)?;

//    let diag_output = DiagnosticOutput;
    
    let num_iterations = 1_000;
    let write_vti = WriteVti;
    let species_dir = root.join("images").join("species_sequence");
    let species_prefix = "species_fields";

    let mut logger = CsvLogger::new(root.join("logs"))?;
                           
    world.mut_time().start_iteration_time();
    while world.time().iteration() < num_iterations {
        // TODO: Where does this belong
        world.mut_time().advance_iteration();

        world.compute_rho(&mut all_species);
    
        world.solve_potential_gs_sor(5000, SorSolverConfig::default()).map_err(anyhow::Error::msg)?;
        world.compute_ef().map_err(anyhow::Error::msg)?;
        for s in all_species.iter_mut() {
            s.advance(&world);
            s.compute_number_density(&world.world_spec());
        }
        
        logger.log(&world, &all_species)?;
        let iter = world.time().iteration(); // TODO: feels a bit sloppy
        if iter % 10 == 0 {
            println!("Iter {iter}");
            write_vti.write_species_at_time_to_vti(&world, &all_species, iter, &species_dir, &species_prefix)?;
            //diag_output.print_status(&world, &all_species);
        }
    }
    
    Ok(())
}
