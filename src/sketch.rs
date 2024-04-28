use std::cmp::max;
use std::path::Path;
use std::time::Instant;

use std::collections::HashSet;

use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use seq_io::fasta::Reader;

use needletail::{parse_fastx_file, Sequence};

// #[inline]
// pub fn mm_hash64(kmer: u64) -> u64 {
//     let mut key = kmer;
//     key = !key.wrapping_add(key << 21); // key = (key << 21) - key - 1;
//     key = key ^ key >> 24;
//     key = (key.wrapping_add(key << 3)).wrapping_add(key << 8); // key * 265
//     key = key ^ key >> 14;
//     key = (key.wrapping_add(key << 2)).wrapping_add(key << 4); // key * 21
//     key = key ^ key >> 28;
//     key = key.wrapping_add(key << 31);
//     key
// }

#[inline]
pub fn mm_hash64(kmer: u64) -> u64 {
    let mut key = kmer;
    key = !key + (key << 21);
    key = key ^ key >> 24;
    key = (key + (key << 3)) + (key << 8);
    key = key ^ key >> 14;
    key = (key + (key << 2)) + (key << 4);
    key = key ^ key >> 28;
    key = key + (key << 31);
    key
}

pub fn sketch(path_fna: &String, ksize: usize, scaled: u64) -> Vec<HashSet<u64>> {
    // get files
    let files: Vec<_> = glob(Path::new(&path_fna).join("*.fna").to_str().unwrap())
        .expect("Failed to read glob pattern")
        .collect();

    // create set
    let n_files = files.len();
    let mut sketch_kmer_sets = Vec::<HashSet<u64>>::with_capacity(n_files);

    // progress bar
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .unwrap()
            .progress_chars("##-"),
    );

    // start sketching
    let threshold = u64::MAX / scaled;
    for i in 0..n_files {
        let mut fastx_reader =
            parse_fastx_file(&files[i].as_ref().unwrap()).expect("Opening .fna files failed");

        // let mut fna_seqs = Vec::<u8>::new();

        sketch_kmer_sets.push(HashSet::<u64>::default());

        while let Some(record) = fastx_reader.next() {
            let seqrec = record.expect("invalid record");

            // let mut seq_i = seqrec.seq().to_vec();
            // seq_i.push(b'N');
            // fna_seqs.append(&mut seq_i);

            let norm_seq = seqrec.normalize(false);

            for (_, (kmer_u64, _), _) in norm_seq.bit_kmers(ksize as u8, true) {
                let h = mm_hash64(kmer_u64);
                if h < threshold {
                    sketch_kmer_sets[i].insert(h);
                }
            }
        }

        pb.inc(1);
        pb.eta();
    }

    pb.finish();
    sketch_kmer_sets

    // let index_vec: Vec<usize> = (0..files.len()).collect();
    // let file_sketch: Vec<Sketch> = index_vec
    //     .par_iter()
    //     .map(|i| {
    //         let file = files[*i].as_ref().unwrap().clone();
    //         let mut sketch = Sketch::new(
    //             String::from(file.file_name().unwrap().to_str().unwrap()),
    //             &params,
    //         );

    //         // Extract kmer hash from genome sequence
    //         extract_kmer_hash(file, &mut sketch);

    //         // Encode extracted kmer hash into sketch HV
    //         if is_x86_feature_detected!("avx2") {
    //             unsafe {
    //                 hd::encode_hash_hd_avx2(&mut sketch);
    //             }
    //         } else {
    //             hd::encode_hash_hd(&mut sketch);
    //         }

    //         // Pre-compute HV's norm
    //         dist::compute_hv_l2_norm(&mut sketch);

    //         // Sketch HV compression
    //         if params.if_compressed {
    //             unsafe {
    //                 hd::compress_hd_sketch(&mut sketch);
    //             }
    //         }

    //         pb.inc(1);
    //         pb.eta();
    //         sketch
    //     })
    //     .collect();

    // pb.finish();

    // println!(
    //     "Sketching {} files took {:.3}\t {:.1} files/s",
    //     files.len(),
    //     pb.elapsed().as_secs_f32(),
    //     (files.len() as f32 / pb.elapsed().as_secs_f32())
    // );
}

use cudarc::driver::{
    CudaDevice, CudaFunction, DeviceRepr, DriverError, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::Ptx;

const CUDA_KERNEL_MY_STRUCT: &str = include_str!(concat!(env!("OUT_DIR"), "/my_struct_kernel.ptx"));

const SEQ_NT4_TABLE: [u8; 256] = [
    0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
];

pub fn sketch_cuda(path_fna: &String, ksize: usize, scaled: u64) -> Vec<HashSet<u64>> {
    // get files
    let files: Vec<_> = glob(Path::new(&path_fna).join("*.fna").to_str().unwrap())
        .expect("Failed to read glob pattern")
        .collect();

    // create set
    let n_files = files.len();
    let mut sketch_kmer_sets = Vec::<HashSet<u64>>::with_capacity(n_files);

    // progress bar
    let pb = ProgressBar::new(files.len() as u64);

    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .unwrap()
            .progress_chars("##-"),
    );

    // setup GPU device
    let now = Instant::now();
    let gpu = CudaDevice::new(0).unwrap();
    println!("Time taken to initialise CUDA: {:.2?}", now.elapsed());

    // compile ptx
    let now = Instant::now();
    let ptx = Ptx::from_src(CUDA_KERNEL_MY_STRUCT);
    gpu.load_ptx(ptx, "my_module", &["my_struct_kernel"])
        .unwrap();
    println!("Time taken to compile and load PTX: {:.2?}", now.elapsed());

    let f = gpu.get_func("my_module", "my_struct_kernel").unwrap();

    // start sketching
    for i in 0..n_files {
        let mut fastx_reader = Reader::from_path(files[i].as_ref().unwrap()).unwrap();

        let mut fna_seqs = Vec::<u8>::new();
        // let mut fna_seqs = Vec::<u8>::with_capacity(10_000_000);

        // let now = Instant::now();

        sketch_kmer_sets.push(HashSet::<u64>::default());
        while let Some(record) = fastx_reader.next() {
            let seqrec = record.unwrap();
            // let seq_i = seqrec.seq();
            // fna_seqs.extend_from_slice(seq_i);

            let mut seq_i = seqrec.owned_seq();
            seq_i.push(b'N');
            fna_seqs.append(&mut seq_i);
        }

        // println!("Time taken to extract seq: {:.2?}", now.elapsed());

        let n_bps = fna_seqs.len();
        let n_kmers = n_bps - ksize + 1;
        let bp_per_thread = 4096;
        let n_threads = (n_kmers + bp_per_thread - 1) / bp_per_thread;
        // println!("nbps={}, nkmers={}, nthreads={}", n_bps, n_kmers, n_threads);

        // copy to GPU
        // let now = Instant::now();

        let gpu_seq = gpu.htod_copy(fna_seqs.clone()).unwrap();
        let gpu_seq_nt4_table = gpu.htod_copy(SEQ_NT4_TABLE.to_vec()).unwrap();
        // allocate 4x more space that expected
        let n_hash_per_thread = (bp_per_thread - ksize + 1) / scaled as usize * 4;
        let n_hash_array = n_hash_per_thread * n_threads;
        let gpu_kmer_bit_packed = gpu.alloc_zeros::<u64>(n_hash_array).unwrap();

        // println!("Time taken to copy to gpu: {:.2?}", now.elapsed());

        // execute kernel
        // let now = Instant::now();

        let config = LaunchConfig::for_num_elems(n_threads as u32);
        let params = (
            &gpu_seq,
            n_bps,
            bp_per_thread,
            n_hash_per_thread,
            ksize,
            u64::MAX / scaled,
            true,
            &gpu_seq_nt4_table,
            &gpu_kmer_bit_packed,
        );
        unsafe { f.clone().launch(config, params) }.unwrap();

        // println!("Time taken to execute kernel: {:.2?}", now.elapsed());

        // gpu.fork_default_stream()
        // gpu.htod_copy(src)
        // unsafe{
        //     f.clone().launch_on_stream(stream, cfg, params)
        // }

        // let now = Instant::now();

        let host_seq = gpu.sync_reclaim(gpu_seq).unwrap();
        let host_kmer_bit_packed = gpu.sync_reclaim(gpu_kmer_bit_packed).unwrap();

        // println!("Time taken to copy from gpu: {:.2?}", now.elapsed());

        // let now = Instant::now();

        for h in host_kmer_bit_packed {
            if h != 0 {
                sketch_kmer_sets[i].insert(h);
            }
        }

        // println!("Time taken to postprocess: {:.2?}", now.elapsed());

        pb.inc(1);
        pb.eta();
    }

    pb.finish();

    sketch_kmer_sets
}

pub fn sketch_cuda_parallel(path_fna: &String, ksize: usize, scaled: u64) -> Vec<HashSet<u64>> {
    // get files
    let files: Vec<_> = glob(Path::new(&path_fna).join("*.fna").to_str().unwrap())
        .expect("Failed to read glob pattern")
        .collect();

    // progress bar
    let pb = ProgressBar::new(files.len() as u64);

    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .unwrap()
            .progress_chars("##-"),
    );

    // setup GPU device
    let now = Instant::now();
    let gpu = CudaDevice::new(0).unwrap();
    println!("Time taken to initialise CUDA: {:.2?}", now.elapsed());

    // compile ptx
    let now = Instant::now();
    let ptx = Ptx::from_src(CUDA_KERNEL_MY_STRUCT);
    gpu.load_ptx(ptx, "my_module", &["my_struct_kernel"])
        .unwrap();
    let gpu = &gpu;
    println!("Time taken to compile and load PTX: {:.2?}", now.elapsed());

    // start sketching
    // Option 1
    let index_vec: Vec<usize> = (0..files.len()).collect();
    let sketch_kmer_sets: Vec<HashSet<u64>> = index_vec
        .par_iter()
        .map(|i| {
            // NOTE: this is the important call to have
            // without this, you'll get a CUDA_ERROR_INVALID_CONTEXT
            gpu.bind_to_thread().unwrap();

            // gpu.fork_default_stream();

            let mut fastx_reader = Reader::from_path(files[*i].as_ref().unwrap()).unwrap();

            // let now = Instant::now();

            let mut fna_seqs = Vec::<u8>::new();
            while let Some(record) = fastx_reader.next() {
                let seqrec = record.unwrap();
                let mut seq_i = seqrec.owned_seq();
                seq_i.push(b'N');
                fna_seqs.append(&mut seq_i);
            }
            // println!("Time taken to extract seq: {:.2?}", now.elapsed());

            let n_bps = fna_seqs.len();
            let n_kmers = n_bps - ksize + 1;
            let bp_per_thread = 1024;
            let n_threads = (n_kmers + bp_per_thread - 1) / bp_per_thread;

            // copy to GPU
            // let now = Instant::now();

            let gpu_seq = gpu.htod_copy(fna_seqs.clone()).unwrap();
            let gpu_seq_nt4_table = gpu.htod_copy(SEQ_NT4_TABLE.to_vec()).unwrap();
            // allocate 4x more space that expected
            let n_hash_per_thread = max(bp_per_thread / scaled as usize * 3, 8);
            let n_hash_array = n_hash_per_thread * n_threads;
            let gpu_kmer_bit_packed = gpu.alloc_zeros::<u64>(n_hash_array).unwrap();

            // println!("Time taken to copy to gpu: {:.2?}", now.elapsed());

            // execute kernel
            // let now = Instant::now();

            let config = LaunchConfig::for_num_elems(n_threads as u32);
            let params = (
                &gpu_seq,
                n_bps,
                bp_per_thread,
                n_hash_per_thread,
                ksize,
                u64::MAX / scaled,
                true,
                &gpu_seq_nt4_table,
                &gpu_kmer_bit_packed,
            );
            let f = gpu.get_func("my_module", "my_struct_kernel").unwrap();
            unsafe { f.clone().launch(config, params) }.unwrap();

            // println!("Time taken to execute kernel: {:.2?}", now.elapsed());

            // let now = Instant::now();

            // let host_seq = gpu.sync_reclaim(gpu_seq).unwrap();
            let host_kmer_bit_packed = gpu.sync_reclaim(gpu_kmer_bit_packed).unwrap();

            // println!("Time taken to copy from gpu: {:.2?}", now.elapsed());

            // let now = Instant::now();

            let mut sketch_kmer_set = HashSet::<u64>::default();
            for h in host_kmer_bit_packed {
                if h != 0 {
                    sketch_kmer_set.insert(h);
                }
            }

            // println!("Time taken to postprocess: {:.2?}", now.elapsed());
            pb.inc(1);
            pb.eta();
            sketch_kmer_set
        })
        .collect();

    pb.finish();

    sketch_kmer_sets
}
