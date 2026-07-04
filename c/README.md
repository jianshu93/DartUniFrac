# DartUniFrac C API

Build the Rust library and C example:

```sh
cd c
make
```

The Makefile selects platform features automatically:

- Linux: `intel-mkl-static,stdsimd`
- macOS: `stdsimd,macos-accelerate`
- other platforms: `stdsimd`

Override the feature set when needed:

```sh
make FEATURES=intel-mkl-static,stdsimd
```

The Rust build produces `target/release/libdartunifrac.so` on Linux,
`target/release/libdartunifrac.dylib` on macOS, and
`target/release/libdartunifrac.a` for static linking.

Minimal usage:

```c
DartUniFracConfig config = dartunifrac_config_default();
config.tree_path = "data/ASVs_aligned.tre";
config.biom_path = "data/ASVs_counts.biom";
config.output_path = "unifrac.tsv";

int code = dartunifrac_run(&config);
if (code != DARTUNIFRAC_OK) {
    fprintf(stderr, "%s\n", dartunifrac_last_error_message());
}
```

Use exactly one of `input_tsv_path` or `biom_path`. Flags are `uint8_t`
booleans where `0` is false and non-zero is true.
