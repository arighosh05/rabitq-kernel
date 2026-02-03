# RaBitQ Kernel

High-performance distance estimation kernel for [RaBitQ](https://github.com/gaoj0017/RaBitQ) binary quantization.

## Performance

| Configuration | ns/vector | Throughput |
|---------------|-----------|------------|
| AVX-512 (4 threads) | 1.89 ns | 529M/s |

Benchmarked on AWS c6i.4xlarge (8 vCPUs / 4 physical cores), 50K vectors, dim=1024.

## Requirements

- CPU with AVX-512F, AVX-512BW, AVX-512VPOPCNTDQ
  - Intel: Ice Lake (2019) or newer
  - AMD: Zen 4 (2022) or newer
- Rust 1.70+

## Quick Start

```bash
git clone https://github.com/arighosh05/rabitq-kernel.git
cd rabitq-kernel

# Run benchmarks
RUSTFLAGS="-C target-cpu=native" cargo bench
```

## Running on AWS EC2

### 1. Launch Instance

```bash
# Instance: c6i.4xlarge (8 vCPUs, 4 physical cores, Ice Lake)
# AMI: Ubuntu 24.04 LTS (or 22.04)
# Storage: 20GB gp3
```

### 2. Connect

```bash
ssh -i your-key.pem ubuntu@<instance-ip>
```

### 3. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### 4. Install C Toolchain

```bash
sudo apt update
sudo apt install -y build-essential
```

### 5. Clone and Build

```bash
git clone https://github.com/arighosh05/rabitq-kernel.git
cd rabitq-kernel

# Build and run benchmarks (use 8 threads for best results)
RAYON_NUM_THREADS=8 RUSTFLAGS="-C target-cpu=native" cargo bench
```

## License

MIT
