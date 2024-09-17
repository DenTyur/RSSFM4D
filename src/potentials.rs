use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

use crate::parameters;
use parameters::Xspace;

pub struct AtomicPotential<'a> {
    pub potential: Array<Complex<f64>, Ix4>,
    x: &'a Xspace,
}

impl<'a> AtomicPotential<'a> {
    pub fn save_potential(&self, path: &str) -> Result<(), WriteNpyError> {
        let writer = BufWriter::new(File::create(path)?);
        self.potential.write_npy(writer)?;
        Ok(())
    }

    pub fn init_from_file(atomic_potential_path: &str, x: &'a Xspace) -> Self {
        // Загружает потенциал из файла.

        let reader = File::open(atomic_potential_path).unwrap();
        Self {
            potential: Array::<Complex<f64>, Ix4>::read_npy(reader).unwrap(),
            x,
        }
    }
}
