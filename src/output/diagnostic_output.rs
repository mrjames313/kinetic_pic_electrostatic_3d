use anyhow::Result;
use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use serde::Serialize;

use crate::world_3d::ThreeDWorldSpec;
use crate::particles::Species;

pub struct SpeciesRow {
    timestep: usize,
    time: f64,
    name: String,
    mp_count: usize,
    real_count: usize,
    momentum_x: f64,
    momentum_y: f64,
    momentum_z: f64,
    kinetic_e: f64,
}
    
#[derive(Serialize)]
pub struct IterRow {
    timestep: usize,
    time: f64,
    wall_time: f64,
    potential_e: f64,
    total_e: f64
}

pub struct CsvLogger {
    wtr: csv::Writer<BufWriter<File>>,
}

impl CsvLogger {
    pub fn new(path: &str) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        let wtr = csv::WriterBuilder::new()
            .has_headers(true)
            .from_writer(BufWriter::new(file));
        Ok(Self {wtr})
    }

    pub fn log(&mut self, row: &IterRow) -> Result<()> {
        self.wtr.serialize(row)?;
        self.wtr.flush()?;
        Ok(())
    }
    
// TODO: finish this logging code
    //    pub fn log_species_rows(&
    
}

pub struct DiagnosticOutput;

impl DiagnosticOutput {

    pub fn print_status(&self, world : &ThreeDWorldSpec, all_species : &Vec<Species>) {
        print!("ts: {}, ", world.get_time());
        for s in all_species.iter() {
            print!("{}: {}", s.name, s.get_num_particles());
        }
        println!("");
    }

}
