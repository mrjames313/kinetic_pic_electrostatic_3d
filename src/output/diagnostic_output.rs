use anyhow::Result;
use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::path::Path;
use serde::Serialize;

use crate::world_3d::ThreeDWorldSpec;
use crate::world_3d::three_d_world_spec::{get_iter_info_from_world, get_time_info_from_world};
use crate::particles::Species;
use crate::particles::particles::get_species_info_from_species;

// TODO: Is making all these pub the right thing?
#[derive(Debug)]
pub struct TimeInfo {
    pub iteration: usize,
    pub sim_time: f64,
    pub wall_time: f64,
}

#[derive(Debug)]
pub struct SpeciesInfo<'a> {
    pub name: &'a str,
    pub mp_count: usize,
    pub real_count: f64,
    pub momentum_x: f64,
    pub momentum_y: f64,
    pub momentum_z: f64,
    pub kinetic_e: f64,
}
    
#[derive(Debug)]
pub struct IterInfo {
    pub potential_e: f64,
    pub total_e: f64
}

// Ugly - thought that serde could flatten these to make a nice struct
#[derive(Debug, Serialize)]
struct SpeciesRow<'a> {
// TimeInfo
    pub iteration: usize,
    pub sim_time: f64,
    pub wall_time: f64,
// SpeciesInfo
    pub name: &'a str,
    pub mp_count: usize,
    pub real_count: f64,
    pub momentum_x: f64,
    pub momentum_y: f64,
    pub momentum_z: f64,
    pub kinetic_e: f64,
}

#[derive(Debug, Serialize)]
struct IterRow {
// TimeInfo
    pub iteration: usize,
    pub sim_time: f64,
    pub wall_time: f64,
// IterInfo
    pub potential_e: f64,
    pub total_e: f64

}

pub struct CsvLogger {
    iter_wtr: csv::Writer<BufWriter<File>>,
    species_wtr: csv::Writer<BufWriter<File>>,
    flush_every: usize,
}

impl CsvLogger {
    pub fn new(out_dir: impl AsRef<Path>) -> Result<Self> {
        std::fs::create_dir_all(out_dir.as_ref())?;
        
        let iter_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(out_dir.as_ref().join("iter.csv"))?;

        let species_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(out_dir.as_ref().join("species.csv"))?;
        
        let iter_wtr = csv::WriterBuilder::new()
            .has_headers(true)
            .from_writer(BufWriter::new(iter_file));

        let species_wtr = csv::WriterBuilder::new()
            .has_headers(true)
            .from_writer(BufWriter::new(species_file));
        
        Ok(Self {iter_wtr, species_wtr, flush_every: 100, })
    }

    pub fn log_species_rows(&mut self, t_info: &TimeInfo, 
                            all_species: &[Species]) -> Result<()> {
        for sp in all_species {
            let s_info = get_species_info_from_species(sp);
            let row: SpeciesRow = SpeciesRow {
                iteration: t_info.iteration,
                sim_time: t_info.sim_time,
                wall_time: t_info.wall_time,
                name: s_info.name,
                mp_count: s_info.mp_count,
                real_count: s_info.real_count,
                momentum_x: s_info.momentum_x,
                momentum_y: s_info.momentum_y,
                momentum_z: s_info.momentum_z,
                kinetic_e: s_info.kinetic_e,
            };
            self.species_wtr.serialize(row)?;

        }
        if t_info.iteration % self.flush_every == 0 {
            self.species_wtr.flush()?;
        }
        Ok(())
    }

    pub fn log_iter_row(&mut self, t_info: &TimeInfo,
                        world: &ThreeDWorldSpec) -> Result<()> {
        let i_info: IterInfo = get_iter_info_from_world(&world);
        let row: IterRow = IterRow{
            iteration: t_info.iteration,
            sim_time: t_info.sim_time,
            wall_time: t_info.wall_time,
            potential_e: i_info.potential_e,
            total_e: i_info.total_e,
        };

        
        if let Err(e) = self.iter_wtr.serialize(&row) {
            eprintln!("CSV serialize failed {e:?}");
            return Err(anyhow::anyhow!("csv serialize fail"));
        }

        if t_info.iteration % self.flush_every == 0 {
            self.iter_wtr.flush()?;
        }
        Ok(())
    }

    
    pub fn log(&mut self, world: &ThreeDWorldSpec,
               all_species: &[Species]) -> Result<()> {
        let t_info = get_time_info_from_world(&world);
        self.log_iter_row(&t_info, &world)?;
        self.log_species_rows(&t_info, &all_species)?;
        Ok(())
    }
    
}

pub struct DiagnosticOutput;

impl DiagnosticOutput {

    pub fn print_status(&self, world : &ThreeDWorldSpec, all_species : &Vec<Species>) {
        print!("ts: {}, ", world.get_sim_time());
        for s in all_species.iter() {
            print!("{}: {}", s.name, s.get_num_particles());
        }
        println!("");
    }

}
