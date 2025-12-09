You might see the following information for old GPU driver versions:

thread 'main' (28055) panicked at /opt/conda/conda-bld/dartunifrac-gpu_1765137471151/_build_env/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cudarc-0.18.1/src/driver/sys/mod.rs:20023:18:
Expected symbol in library: DlSym { desc: "/lib64/libcuda.so: undefined symbol: cuCtxGetDevice_v2" }
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

Please install the most recent GPU drivers if you see this. 
