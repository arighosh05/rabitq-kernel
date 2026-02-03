use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use rand::Rng;
use rabitq::{BatchData, Query, compute_batch, compute_batch_parallel, is_supported};

fn random_batch(n: usize) -> BatchData {
    let mut rng = rand::thread_rng();
    let mut data = BatchData::new(n);
    
    for x in data.bits.as_mut_slice().iter_mut() {
        *x = rng.gen();
    }
    for x in data.inv_norm.iter_mut() {
        *x = rng.gen_range(0.5..2.0);
    }
    for x in data.pop_term.iter_mut() {
        *x = rng.gen_range(0.0..100.0);
    }
    
    data
}

fn random_query() -> Query {
    let mut rng = rand::thread_rng();
    let mut q = Query::new();
    
    for x in q.bp0.iter_mut() { *x = rng.gen(); }
    for x in q.bp1.iter_mut() { *x = rng.gen(); }
    for x in q.bp2.iter_mut() { *x = rng.gen(); }
    for x in q.bp3.iter_mut() { *x = rng.gen(); }
    
    q.factor_ip = rng.gen();
    q.factor_pop = rng.gen();
    q.constant_term = rng.gen();
    
    q
}

fn bench_50k(c: &mut Criterion) {
    if !is_supported() {
        println!("AVX-512 VPOPCNTDQ not supported");
        return;
    }

    let mut group = c.benchmark_group("rabitq_50k");
    
    let data = random_batch(50_000);
    let query = random_query();
    let mut results = vec![0u32; 50_000];
    
    group.throughput(Throughput::Elements(50_000));
    
    group.bench_function("compute_batch", |b| {
        b.iter(|| unsafe {
            compute_batch(
                black_box(&data),
                black_box(&query),
                black_box(&mut results),
            )
        })
    });
    
    group.finish();
}

fn bench_scaling(c: &mut Criterion) {
    if !is_supported() {
        return;
    }

    let mut group = c.benchmark_group("rabitq_scaling");
    
    for size in [1_000, 10_000, 50_000, 100_000] {
        let data = random_batch(size);
        let query = random_query();
        let mut results = vec![0u32; size];
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, _| {
                b.iter(|| unsafe {
                    compute_batch(
                        black_box(&data),
                        black_box(&query),
                        black_box(&mut results),
                    )
                })
            },
        );
    }
    
    group.finish();
}

fn bench_parallel(c: &mut Criterion) {
    if !is_supported() {
        return;
    }

    let mut group = c.benchmark_group("parallel_50k");
    
    let data = random_batch(50_000);
    let query = random_query();
    let mut results = vec![0u32; 50_000];
    
    group.throughput(Throughput::Elements(50_000));
    
    // Single-threaded baseline
    group.bench_function("single_threaded", |b| {
        b.iter(|| unsafe {
            compute_batch(
                black_box(&data),
                black_box(&query),
                black_box(&mut results),
            )
        })
    });
    
    // Parallel with global pool (default thread count = num CPUs)
    group.bench_function("parallel_auto", |b| {
        b.iter(|| {
            compute_batch_parallel(
                black_box(&data),
                black_box(&query),
                black_box(&mut results),
            )
        })
    });
    
    group.finish();
}

criterion_group!(benches, bench_50k, bench_scaling, bench_parallel);
criterion_main!(benches);
