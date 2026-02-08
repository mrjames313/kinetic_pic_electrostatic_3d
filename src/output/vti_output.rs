use std::path::{Path};
use vtkio::model::{Attribute, Attributes, ByteOrder, DataSet, Extent,
                   ImageDataPiece, Piece, Version, Vtk};

use crate::particles::Species;
use crate::world_3d::ThreeDWorld;


pub struct WriteVti;

impl WriteVti {
    // Will append the timestep and .vti suffix to the given path
    
    pub fn write_species_at_time_to_vti(&self, world : &ThreeDWorld,
                                        all_species : &Vec<Species>,
                                        timestep : usize,
                                        out_dir: impl AsRef<Path>,
                                        prefix: &str)
                                        -> anyhow::Result<()> {

        // construct the new name
        let out_file = out_dir.as_ref().join(format!("{prefix}_{timestep:09}.vti"));
//        println!("Writing to file {}", out_file.display());
        
        let npts = world.world_spec().x_dim().n() * world.world_spec().y_dim().n() * world.world_spec().z_dim().n();

        // Attach arrays as POINT_DATA (common for potentials/fields sampled at grid points).
        let mut attrs = Attributes::new();

        for species in all_species.iter() {
            anyhow::ensure!(species.number_density.len() == npts,
                            "{} density: wrong length", species.name);
            attrs.point.push(Attribute::scalars(species.name.clone(), 1)
                             .with_data(species.number_density.data().to_vec() ));
        }

        // ImageData uses an extent + origin + spacing. Extent::Dims is the legacy “dims” form. :contentReference[oaicite:5]{index=5}
        // ImageData uses inclusive ranges, so 0..20 includes 21 points.
        let extent = Extent::Ranges([0..=((world.world_spec().x_dim().n() as i32) - 1), 0..=((world.world_spec().y_dim().n() as i32) - 1), 0..=((world.world_spec().z_dim().n() as i32) - 1) ]);

        let piece = ImageDataPiece {
            extent: extent.clone(),
            data: attrs,
        };

        let vtk = Vtk {
            version: Version::new((2, 3)),
            byte_order: ByteOrder::LittleEndian,
            title: "All Species snapshot".to_string(),
            file_path: None,
            data: DataSet::ImageData {
                extent,
                origin: [world.world_spec().x_dim().min() as f32, world.world_spec().y_dim().min() as f32, world.world_spec().z_dim().min() as f32],
                spacing: [world.world_spec().x_dim().delta() as f32, world.world_spec().y_dim().delta() as f32, world.world_spec().z_dim().delta() as f32],
                meta: None,
                pieces: vec![Piece::Inline(Box::new(piece))],
            },
        };

        vtk.export(out_file)?;
        Ok(())
    }
}
