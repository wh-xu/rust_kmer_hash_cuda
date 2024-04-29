#![allow(non_snake_case)]

use std::str::FromStr;
use std::time::Instant;

mod fastx_reader;
mod sketch_cuda;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build_global()
        .unwrap();

    let path_fna = String::from_str("../../genome-HD/dna-dataset/D1").unwrap();
    // let path_fna = String::from_str("./test").unwrap();
    let ksize = 21;
    let scaled = 2000;

    // fastx_reader::parse_fasta_file(&path_fna, ksize, scaled);
    // fastx_reader::parse_my_fasta_file(&path_fna, ksize, scaled);
    // return Ok(());

    // CUDA parallel
    let now = Instant::now();
    let sketch_hash_gpu_par = sketch_cuda::sketch_cuda_parallel(&path_fna, ksize, scaled);
    println!(
        "Time taken to call sketch on {} : {:.2?}",
        "gpu-parallel",
        now.elapsed()
    );

    // CPU parallel
    let now = Instant::now();
    let sketch_hash_cpu = sketch_cuda::sketch_cpu_parallel(&path_fna, ksize, scaled);
    println!(
        "Time taken to call sketch on {} : {:.2?}",
        "cpu-parallel",
        now.elapsed()
    );

    assert_eq!(sketch_hash_cpu.len(), sketch_hash_gpu_par.len());
    for i in 0..sketch_hash_cpu.len() {
        // println!(
        //     "{}: {}, {}",
        //     i,
        //     sketch_hash_cpu[i].len(),
        //     sketch_hash_gpu_par[i].len()
        // );

        // assert_eq!(sketch_hash_cpu[i].len(), sketch_hash_gpu_par[i].len());

        let overlap = sketch_hash_cpu[i].intersection(&sketch_hash_gpu_par[i]);
        let overlap = overlap.count() as f32 / sketch_hash_cpu[i].len() as f32;
        assert!(overlap > 0.999, "overlap = {:.4?}", overlap);
    }

    // // CUDA
    // let now = Instant::now();
    // let sketch_hash_gpu = sketch::sketch_cuda(&path_fna, ksize, scaled);
    // println!(
    //     "Time taken to call sketch on {} : {:.2?}",
    //     "gpu",
    //     now.elapsed()
    // );

    // // CPU
    // let now = Instant::now();
    // let sketch_hash_cpu = sketch::sketch_cpu(&path_fna, ksize, scaled);
    // println!(
    //     "Time taken to call sketch on {} : {:.2?}",
    //     "cpu",
    //     now.elapsed()
    // );

    // assert_eq!(sketch_hash_cpu.len(), sketch_hash_gpu.len());
    // for i in 0..sketch_hash_cpu.len() {
    //     let mut vec_cpu: Vec<u64> = sketch_hash_cpu[i].clone().into_iter().collect();
    //     vec_cpu.sort();
    //     let mut vec_gpu: Vec<u64> = sketch_hash_gpu[i].clone().into_iter().collect();
    //     vec_gpu.sort();
    // }
}
