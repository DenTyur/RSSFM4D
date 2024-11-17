#[macro_use]
extern crate fstrings;

mod evolution;
mod field;
mod parameters;
mod potentials;
mod wave_function;
use evolution::{time_step_evol, FftMaker4d};
use field::Field4D;
use parameters::{Pspace, Tspace, Xspace};
use potentials::AtomicPotential;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Error, Write};
use std::time::Instant;
use wave_function::WaveFunction;

fn main() -> Result<(), Error> {
    catalogs_check();
    // пути к сохраненным массивам
    let x_dir_path = "arrays_saved";
    let atomic_potential_path = "arrays_saved/atomic_potential.npy";
    let psi_path = "arrays_saved/psi_initial.npy";

    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.2, 10, 5);
    t.save_grid("arrays_saved/time_evol/t.npy").unwrap();

    // задаем координатную сетку
    let x = Xspace::load(x_dir_path, 4);

    // инициализируем импульсную сетку на основе координатной сетки
    let p = Pspace::init(&x);

    // инициализируем внешнее поле
    let field1d = Field4D {
        amplitude: 0.0,
        omega: 0.002,
        x_envelop: 50.0001,
    };

    // генерация "атомного" потенциала
    let atomic_potential = AtomicPotential::init_from_file(atomic_potential_path, &x);

    // генерация начальной волновой функции psi
    let mut psi = WaveFunction::init_from_file(psi_path, &x);
    psi.normalization_by_1();

    // планировщик fft
    let mut fft = FftMaker4d::new(&x.n);

    // файл для сохранения сообщений
    let path = "main_out.txt";
    let mut output = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .unwrap();

    write!(output, "=========================\n")?;
    write!(output, "---------- RUN ----------\n")?;
    write!(output, "=========================\n")?;

    let total_time = Instant::now();
    for i in 0..t.nt {
        // сохранение временного среза волновой функции
        psi.save_psi(f!("arrays_saved/time_evol/psi_x/psi_t_{i}.npy").as_str())
            .unwrap();
        // эволюция на 1 шаг (t_step) по времени
        let time = Instant::now();
        time_step_evol(
            &mut fft,
            &mut psi,
            &field1d,
            &atomic_potential,
            &x,
            &p,
            &mut t,
        );
        println!(
            "STEP {}/{}  time_step = {:.5}  total_time = {:.5}",
            i,
            t.nt,
            time.elapsed().as_secs_f32(),
            total_time.elapsed().as_secs_f32()
        );
        println!("t.current={:.5}, norm = {}", t.current, psi.norm());
        write!(
            output,
            "STEP {}/{}  time_step = {:.5}  total_time = {:.5}\n",
            i,
            t.nt,
            time.elapsed().as_secs_f32(),
            total_time.elapsed().as_secs_f32()
        )?;
        write!(
            output,
            "t.current={:.5}, norm = {}\n",
            t.current,
            psi.norm()
        )?;
    }
    Ok(())
}

fn catalogs_check() {
    use std::fs;
    use std::path::Path;
    let paths = [
        "arrays_saved",
        "arrays_saved/time_evol",
        "arrays_saved/time_evol/psi_x",
    ];
    for path in paths {
        if !Path::new(path).exists() {
            fs::create_dir(path).unwrap();
        }
    }
}
