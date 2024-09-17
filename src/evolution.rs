use crate::field;
use crate::parameters;
use crate::potentials;
use crate::wave_function;
use field::Field4D;
use itertools::multizip;
use ndarray::prelude::*;
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use num_complex::Complex;
use parameters::*;
use potentials::AtomicPotential;
use rayon::prelude::*;
use std::f64::consts::PI;
use wave_function::WaveFunction;

pub fn time_step_evol(
    fft: &mut FftMaker4d,
    psi: &mut WaveFunction,
    field: &Field4D,
    u: &AtomicPotential,
    x: &Xspace,
    p: &Pspace,
    t: &mut Tspace,
) {
    modify_psi(psi, x, p);
    x_evol_half(psi, u, t, field, x);

    for _i in 0..t.n_steps - 1 {
        fft.do_fft(psi);
        // Можно оптимизировать p_evol
        p_evol(psi, p, t.dt);
        fft.do_ifft(psi);
        x_evol(psi, u, t, field, x);
        t.current += t.dt;
    }

    fft.do_fft(psi);
    p_evol(psi, p, t.dt);
    fft.do_ifft(psi);
    x_evol_half(psi, u, t, field, x);
    demodify_psi(psi, x, p);
    t.current += t.dt;
}

pub fn x_evol_half(
    psi: &mut WaveFunction,
    atomic_potential: &AtomicPotential,
    t: &Tspace,
    field: &Field4D,
    x: &Xspace,
) {
    // эволюция в координатном пространстве на половину временного шага
    let j = Complex::I;

    let electric_field_potential = field.potential_as_array(t.current, x);

    multizip((
        psi.psi.axis_iter_mut(Axis(0)),
        atomic_potential.potential.axis_iter(Axis(0)),
        electric_field_potential[0].iter(),
    ))
    .par_bridge()
    .for_each(|(mut psi_123, atomic_potential_123, field_potential_0ax)| {
        multizip((
            psi_123.axis_iter_mut(Axis(0)),
            atomic_potential_123.axis_iter(Axis(0)),
            electric_field_potential[1].iter(),
        ))
        .for_each(|(mut psi_23, atomic_potential_23, field_potential_1ax)| {
            multizip((
                psi_23.axis_iter_mut(Axis(0)),
                atomic_potential_23.axis_iter(Axis(0)),
                electric_field_potential[2].iter(),
            ))
            .for_each(|(mut psi_3, atomic_potential_3, field_potential_2ax)| {
                multizip((
                    psi_3.iter_mut(),
                    atomic_potential_3.iter(),
                    electric_field_potential[3].iter(),
                ))
                .for_each(
                    |(psi_elem, atomic_potential_elem, field_potential_3ax)| {
                        *psi_elem *= (-j
                            * 0.5
                            * t.dt
                            * (atomic_potential_elem
                                - field_potential_0ax
                                - field_potential_1ax
                                - field_potential_2ax
                                - field_potential_3ax))
                            .exp()
                    },
                )
            })
        })
    });
}

pub fn x_evol(
    psi: &mut WaveFunction,
    atomic_potential: &AtomicPotential,
    t: &Tspace,
    field: &Field4D,
    x: &Xspace,
) {
    // эволюция в координатном пространстве на половину временного шага
    let j = Complex::I;

    let electric_field_potential = field.potential_as_array(t.current, x);

    multizip((
        psi.psi.axis_iter_mut(Axis(0)),
        atomic_potential.potential.axis_iter(Axis(0)),
        electric_field_potential[0].iter(),
    ))
    .par_bridge()
    .for_each(|(mut psi_123, atomic_potential_123, field_potential_0ax)| {
        multizip((
            psi_123.axis_iter_mut(Axis(0)),
            atomic_potential_123.axis_iter(Axis(0)),
            electric_field_potential[1].iter(),
        ))
        .for_each(|(mut psi_23, atomic_potential_23, field_potential_1ax)| {
            multizip((
                psi_23.axis_iter_mut(Axis(0)),
                atomic_potential_23.axis_iter(Axis(0)),
                electric_field_potential[2].iter(),
            ))
            .for_each(|(mut psi_3, atomic_potential_3, field_potential_2ax)| {
                multizip((
                    psi_3.iter_mut(),
                    atomic_potential_3.iter(),
                    electric_field_potential[3].iter(),
                ))
                .for_each(
                    |(psi_elem, atomic_potential_elem, field_potential_3ax)| {
                        *psi_elem *= (-j
                            * t.dt
                            * (atomic_potential_elem
                                - field_potential_0ax
                                - field_potential_1ax
                                - field_potential_2ax
                                - field_potential_3ax))
                            .exp()
                    },
                )
            })
        })
    });
}

pub fn p_evol(psi: &mut WaveFunction, p: &Pspace, dt: f64) {
    // эволюция в импульсном пространстве
    let j = Complex::I;
    psi.psi
        .axis_iter_mut(Axis(0))
        .zip(p.grid[0].iter())
        .par_bridge()
        .for_each(|(mut psi_123, p0_i)| {
            psi_123
                .axis_iter_mut(Axis(0))
                .zip(p.grid[1].iter())
                .for_each(|(mut psi_23, p1_i)| {
                    psi_23
                        .axis_iter_mut(Axis(0))
                        .zip(p.grid[2].iter())
                        .for_each(|(mut psi_3, p2_i)| {
                            psi_3
                                .iter_mut()
                                .zip(p.grid[3].iter())
                                .for_each(|(psi_elem, p3_i)| {
                                    *psi_elem *= (-j
                                        * 0.5
                                        * dt
                                        * (p0_i.powi(2)
                                            + p1_i.powi(2)
                                            + p2_i.powi(2)
                                            + p3_i.powi(2)))
                                    .exp()
                                })
                        })
                })
        });
}

pub fn demodify_psi(psi: &mut WaveFunction, x: &Xspace, p: &Pspace) {
    // демодифицирует "psi для DFT" обратно в psi
    let j = Complex::I;
    psi.psi
        .axis_iter_mut(Axis(0))
        .zip(x.grid[0].iter())
        .par_bridge()
        .for_each(|(mut psi_123, x0_i)| {
            psi_123
                .axis_iter_mut(Axis(0))
                .zip(x.grid[1].iter())
                .for_each(|(mut psi_23, x1_i)| {
                    psi_23
                        .axis_iter_mut(Axis(0))
                        .zip(x.grid[2].iter())
                        .for_each(|(mut psi_3, x2_i)| {
                            psi_3
                                .iter_mut()
                                .zip(x.grid[3].iter())
                                .for_each(|(psi_elem, x3_i)| {
                                    *psi_elem *= (2. * PI).powi(2)
                                        / (x.dx[0] * x.dx[1] * x.dx[2] * x.dx[3])
                                        * (j * (p.p0[0] * x0_i
                                            + p.p0[1] * x1_i
                                            + p.p0[2] * x2_i
                                            + p.p0[3] * x3_i))
                                            .exp()
                                })
                        })
                })
        });
}

pub fn modify_psi(psi: &mut WaveFunction, x: &Xspace, p: &Pspace) {
    // модифицирует psi для DFT (в нашем сучае FFT)
    let j = Complex::I;

    psi.psi
        .axis_iter_mut(Axis(0))
        .zip(x.grid[0].iter())
        .par_bridge()
        .for_each(|(mut psi_123, x0_i)| {
            psi_123
                .axis_iter_mut(Axis(0))
                .zip(x.grid[1].iter())
                .for_each(|(mut psi_23, x1_i)| {
                    psi_23
                        .axis_iter_mut(Axis(0))
                        .zip(x.grid[2].iter())
                        .for_each(|(mut psi_3, x2_i)| {
                            psi_3
                                .iter_mut()
                                .zip(x.grid[3].iter())
                                .for_each(|(psi_elem, x3_i)| {
                                    *psi_elem *= (x.dx[0] * x.dx[1] * x.dx[2] * x.dx[3])
                                        / (2. * PI).powi(2)
                                        * (-j
                                            * (p.p0[0] * x0_i
                                                + p.p0[1] * x1_i
                                                + p.p0[2] * x2_i
                                                + p.p0[3] * x3_i))
                                            .exp()
                                })
                        })
                })
        });
}

pub struct FftMaker4d {
    pub handler: Vec<FftHandler<f64>>,
    pub psi_temp: Array4<Complex<f64>>,
}

impl FftMaker4d {
    pub fn new(n: &Vec<usize>) -> Self {
        Self {
            handler: Vec::from_iter(0..n.len()) // тоже костыль! как это сделать через функцию?
                .iter()
                .map(|&i| FftHandler::new(n[i]))
                .collect(),
            psi_temp: Array::zeros((n[0], n[1], n[2], n[3])),
        }
    }

    pub fn do_fft(&mut self, psi: &mut WaveFunction) {
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        ndfft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[1], 1);
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[2], 2);
        ndfft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[3], 3);
    }
    pub fn do_ifft(&mut self, psi: &mut WaveFunction) {
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[1], 1);
        ndifft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[0], 0);
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[2], 2);
        ndifft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[3], 3);
    }
}
