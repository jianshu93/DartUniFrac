<div align="center">
  <img width="30%" src ="DartUniFrac_logo.png">
</div>

# DartUniFrac: Approximate unweighted UniFrac via Weighted MinHash
This crate provides an efficient implementation of the newly invented DartUniFrac algorithm for large-scale [UniFrac](https://en.wikipedia.org/wiki/UniFrac) computation. We named this new algorithm DartUniFrac because the key step is to use DartMinHash or Efficient Rejection Sampling (or ERS) on branches and the DartMinHash/ERS is about "Among the first r darts thrown, return those hitting $x_i$". 

## Overview
UniFrac can be simply described as unique branches that differ two samples over shared branches (see original UniFrac paper [here](https://journals.asm.org/doi/full/10.1128/aem.71.12.8228-8235.2005)). Here, each sample has some taxa (or features) that are in the phylogenetic tree. 

<div align="center">
  <img width="40%" src ="unweighted_unifrac_schematic.png">
</div>

If we reformulate UniFrac in math, it can be descriped below:

$$D_{UniFrac}(A,B) = \frac{\displaystyle \sum_{i\in E} \ell_i \cdot |\max_{j\in {Desc}(i)} x_j(A) - \max_{j\in {Desc}(i)} x_j(B)|}{\displaystyle \sum_{i\in E} \ell_i \cdot \max(\max_{j\in {Desc}(i)} x_j(A), \max_{j\in {Desc}(i)} x_j(B) )}$$
$$x_j(S) =
\begin{cases}
      1, & \text{if taxon } j \text{ is present in sample } S,\\
      0, & \text{otherwise.}
    \end{cases}$$
where Desc(i) are all the descendents of branch i. 
since:
$$\displaystyle \sum_{i\in E} \ell_i \cdot |\max_{j\in {Desc}(i)} x_j(A) - \max_{j\in {Desc}(i)} x_j(B)| = \displaystyle \sum_{i\in E} \ell_i \cdot \max(\max_{j\in {Desc}(i)} x_j(A), \max_{j\in {Desc}(i)} x_j(B)) - \displaystyle \sum_{i\in E} \ell_i \cdot \min(\max_{j\in {Desc}(i)} x_j(A), \max_{j\in {Desc}(i)} x_j(B)) $$

Therefore:

$$D_{UniFrac}(A,B)=1-\frac{\displaystyle \sum_{i\in E} \ell_i \cdot \min(\max_{j\in {Desc}(i)} x_j(A), \max_{j\in {Desc}(i)} x_j(B))}{\displaystyle \sum_{i\in E} \ell_i \cdot \max(\max_{j\in {Desc}(i)} x_j(A), \max_{j\in {Desc}(i)} x_j(B) )}$$

since $\displaystyle \ell_i$ can be moved inside max and min (same for sample A and B) and $\max_{j\in {Desc}(i)} x_j(A)$ and $\max_{j\in {Desc}(i)} x_j(B)$ are either 1 or 0. Therefore, it can be rewritten as:
$$D_{UniFrac}(x,y)=1-J_w(x,y) = \frac{\sum_{i=1}^n \min(x_i, y_i)}{\sum_{i=1}^n \max(x_i, y_i)}$$

here, $\displaystyle J_w(x,y)$ is ***Weighted Jaccard Similarity***, which can be efficiently estimated via Weighted MinHash, a sketching algorithm that is widely used for large-scale text mining. We chose [DartMinHash](https://arxiv.org/abs/2005.11547) and [Efficient Rejection Sampling](https://ojs.aaai.org/index.php/AAAI/article/view/16543) due to their speed for sparse and dense data, respectively. 

In summary, unweighted UniFrac distance can be considered as weighted Jaccard distance on branches. 

## Libraries
We first created a few libraries for the best performance of DartUniFrac implementation. 

1.Optimal representation of balanced parenthesis for phylogenetic trees via [succparen](https://github.com/sile/succparen)

2.Implementation of DartMinHash and Efficient Rejection Sampling algorithms can be found [here](https://github.com/jianshu93/dartminhash-rs).

3.SIMD-aware Hamming similarity for computing hash collision probability of sketches, [anndists](https://github.com/jianshu93/anndists)

## Install
### Pre-compiled on Linux
```bash
wget https://github.com/jianshu93/DartUniFrac/releases/download/v0.1.0/dartunifrac_Linux_x86-64_v0.1.0.zip
unzip dartunifrac_Linux_x86-64_v0.1.0.zip
chmod a+x ./dartunifrac
./dartunifrac -h
```

### macOS via Homebrew: 
```bash
## install homebrew first: https://brew.sh
brew tap jianshu93/DartUniFrac
brew install DartUniFrac
dartunifrac -h
```

### from source
HDF5 needs to be installed first, see guidance [here](hdf5_install.md)
```bash
git clone https://github.com/jianshu93/DartUniFrac.git
cd DartUniFrac
#### You must have HDF5 installed and its library in system path. This is for BIOM format input.
cargo build --release
```

## Usage
DartUniFrac will use all availble CPU cores/threads via Rayon by default.
```bash
$ ./target/release/dartunifrac -h

 ************** initializing logger *****************

Approximate unweighted UniFrac via Weighted MinHash

Usage: dartunifrac [OPTIONS] --tree <tree> <--input <input>|--biom <biom>>

Options:
  -t, --tree <tree>           Input tree in Newick format
  -i, --input <input>         OTU/Feature table in TSV format
  -b, --biom <biom>           OTU/Feature table in BIOM (HDF5) format
  -o, --output <output>       Output distance matrix in TSV format [default: unifrac.tsv]
  -s, --sketch <sketch-size>  Sketch size for Weighted MinHash (DartMinHash or ERS) [default: 1024]
  -m, --method <method>       Sketching method: dmh (DartMinHash) or ers (Efficient Rejection Sampling) [default: dmh] [possible values: dmh, ers]
  -l, --length <seq-length>   Per-hash independent random sequence length L for ERS [default: 4096]
      --seed <seed>           Random seed for reproducibility [default: 1337]
  -h, --help                  Print help
  -V, --version               Print version
```


```bash
### DartMinHash
dartunifrac -t ./data/ASVs_aligned.tre -i ./data/ASVs_counts.txt -m dmh -s 2048 -o unifrac_dmh.csv

### Efficient Rejection Sampling
dartunifrac -t ./data/ASVs_aligned.tre -i ./data/ASVs_counts.txt -m ers -s 2048 -l 4096 -o unifrac_ers.csv
```
## Benchmark
We use Striped UniFrac algorithm as the ground truth, which is an exact and efficient algorithm for large number of samples. A pure Rust implementaion, as a supporting crate for this one, can be found [here](https://github.com/jianshu93/unifrac_bp), also included as a binary in this crate.

For the testing data (ASVs_count.tsv and ASV_aligned.tre), the truth from Striped UniFrac is:

|    | Orwoll_BI0023_BI | Orwoll_BI0056_BI | Orwoll_BI0131_BI | Orwoll_BI0153_BI | Orwoll_BI0215_BI | Orwoll_BI0353_BI |
|---|---:|---:|---:|---:|---:|---:|
| Orwoll_BI0023_BI | 0 | 0.403847873210907 | 0.3646169304847717 | 0.366204708814621 | 0.3484474122524261 | 0.5317433476448059 |
| Orwoll_BI0056_BI | 0.403847873210907 | 0 | 0.3883068859577179 | 0.4069649279117584 | 0.3338068425655365 | 0.5172212719917297 |
| Orwoll_BI0131_BI | 0.3646169304847717 | 0.3883068859577179 | 0 | 0.4222687482833862 | 0.4006152451038361 | 0.5693299174308777 |
| Orwoll_BI0153_BI | 0.366204708814621 | 0.4069649279117584 | 0.4222687482833862 | 0 | 0.2608364820480347 | 0.4297698140144348 |
| Orwoll_BI0215_BI | 0.3484474122524261 | 0.3338068425655365 | 0.4006152451038361 | 0.2608364820480347 | 0 | 0.4572696685791016 |
| Orwoll_BI0353_BI | 0.5317433476448059 | 0.5172212719917297 | 0.5693299174308777 | 0.4297698140144348 | 0.4572696685791016 | 0 |

DartUniFrac estimation (DartMinHash) is: 

|    | Orwoll_BI0023_BI | Orwoll_BI0056_BI | Orwoll_BI0131_BI | Orwoll_BI0153_BI | Orwoll_BI0215_BI | Orwoll_BI0353_BI |
|:--|---:|---:|---:|---:|---:|---:|
| Orwoll_BI0023_BI | 0.0 | 0.40307617187 | 0.36596679687 | 0.36010742187 | 0.35815429687 | 0.530273437 |
| Orwoll_BI0056_BI | 0.40307617187 | 0.0 | 0.3901367187 | 0.40747070312 | 0.3398437 | 0.5209960937 |
| Orwoll_BI0131_BI | 0.36596679687 | 0.3901367187 | 0.0 | 0.41577148437 | 0.40014648437 | 0.56860351562 |
| Orwoll_BI0153_BI | 0.36010742187 | 0.40747070312 | 0.41577148437 | 0.0 | 0.260742187 | 0.422851562 |
| Orwoll_BI0215_BI | 0.35815429687 | 0.3398437 | 0.40014648437 | 0.260742187 | 0.0 | 0.45825195312 |
| Orwoll_BI0353_BI | 0.530273437 | 0.5209960937 | 0.56860351562 | 0.422851562 | 0.45825195312 | 0.0 |

DartUniFrac estimation (Efficient Rejection Sampling) is: 
|    | Orwoll_BI0023_BI | Orwoll_BI0056_BI | Orwoll_BI0131_BI | Orwoll_BI0153_BI | Orwoll_BI0215_BI | Orwoll_BI0353_BI |
|:--|---:|---:|---:|---:|---:|---:|
| Orwoll_BI0023_BI | 0.0 | 0.4101562 | 0.3657226562 | 0.3618164062 | 0.3569335937 | 0.52539062 |
| Orwoll_BI0056_BI | 0.4101562 | 0.0 | 0.40039062 | 0.4086914062 | 0.340820312 | 0.5190429687 |
| Orwoll_BI0131_BI | 0.3657226562 | 0.40039062 | 0.0 | 0.4360351562 | 0.4145507812 | 0.570312 |
| Orwoll_BI0153_BI | 0.3618164062 | 0.4086914062 | 0.4360351562 | 0.0 | 0.264648437 | 0.4311523437 |
| Orwoll_BI0215_BI | 0.3569335937 | 0.340820312 | 0.4145507812 | 0.264648437 | 0.0 | 0.460937 |
| Orwoll_BI0353_BI | 0.52539062 | 0.5190429687 | 0.570312 | 0.4311523437 | 0.460937 | 0.0 |


## Choosing L for Efficent Rejection Sampling (ERS)
The best L for achiving a given accuracy is related to the sparsity of the data (see ERS paper [here](https://ojs.aaai.org/index.php/AAAI/article/view/16543)). The author recommended an equation for L: $l=\frac{\alpha}{s}$, where s is the sparsity of the data while $\alpha$ is a constant, normally 0.5 to 5. If you have a large dataset, you can randomly choose several samples to check the sparsity (relevant branches). You can obtain $\alpha$ using the Striped UniFrac binary: 

```bash
RUST_LOG=info ./target/release/striped_unifrac -t ./data/ASVs_aligned.tre -i ./data/AS
Vs_counts.txt -o ASVs_striped_unifac_dist.tsv
```
You will see some log like this:
```bash
 ************** initializing logger *****************

[2025-08-21T20:12:38Z INFO  striped_unifrac] logger initialized from default environment
[2025-08-21T20:12:38Z INFO  striped_unifrac] Total branches with positive length: 923
[2025-08-21T20:12:38Z INFO  striped_unifrac] Start parsing input.
[2025-08-21T20:12:38Z INFO  striped_unifrac] phase-1 masks built      0 ms
[2025-08-21T20:12:38Z INFO  striped_unifrac] sample 0: relevant branches = 689 / 923
[2025-08-21T20:12:38Z INFO  striped_unifrac] sample 1: relevant branches = 594 / 923
[2025-08-21T20:12:38Z INFO  striped_unifrac] sample 2: relevant branches = 646 / 923
[2025-08-21T20:12:38Z INFO  striped_unifrac] sample 3: relevant branches = 584 / 923
[2025-08-21T20:12:38Z INFO  striped_unifrac] sample 4: relevant branches = 647 / 923
[2025-08-21T20:12:38Z INFO  striped_unifrac] sample 5: relevant branches = 468 / 923
[2025-08-21T20:12:38Z INFO  striped_unifrac] phase-2 sparse lists built (1 strips)
[2025-08-21T20:12:38Z INFO  striped_unifrac] phase-3 block pass      0 ms
[2025-08-21T20:12:38Z INFO  striped_unifrac] Start writing output.

```
This is a dense example where $\alpha$ is almost 75% so L can be small. For real-world datasets, $\alpha$ can be as small as 0.001.


## Acknowledgements
We want to thank [Otmar Ertl](https://www.dynatrace.com/engineering/persons/otmar-ertl/) and [Xiaoyun Li](https://lixiaoyun0239.github.io/cv/) for their helpful comments on DartMinHash and Efficient Rejection Sampling, respectively. We want to thank Yuhan(Sherlyn) Weng for helping with DartUniFrac logo design.

## References

Paper to come