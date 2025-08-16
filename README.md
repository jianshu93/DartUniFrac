<div align="center">
  <img width="30%" src ="DartUniFrac_logo.png">
</div>

# DartUniFrac: Approximate unweighted UniFrac via Weighted MinHash
This crate provides an efficieint implementation of the DartUniFrac algorithm for large-scale UniFrac computation. 

## Overview
UniFrac can be simply described as unique branches that differ two samples over shared branches. Here, each sample has some taxa (or features) that are in the phylogenetic tree. 

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

here, $\displaystyle J_w(x,y)$ is ***Weighted Jaccard Similarity***, which can be efficiently estimated via Weighted MinHash, a sketching algorithm that is widely used for large-scale text mining. We chose [DartMinHash](https://arxiv.org/abs/2005.11547) and [Efficient Rejection Sampling](https://ojs.aaai.org/index.php/AAAI/article/view/16543) due to their speed for sparse and dense data respectively. The library implementation of the algorithms can be found [here](https://github.com/jianshu93/dartminhash-rs).

In summary, unweighted UniFrac distance can be considered as weighted Jaccard distance on branches. 

## Install
```bash
git clone https://github.com/jianshu93/DartUniFrac.git
cd DartUniFrac
#### You must have HDF5 installed and its library in system path. This is for BIOM format input.
cargo build --release
```

## Usage
```bash
Approximate unweighted UniFrac via Weighted MinHash

Usage: dartunifrac [OPTIONS] --tree <tree> <--input <input>|--biom <biom>>

Options:
  -t, --tree <tree>           Input tree in Newick format
  -i, --input <input>         OTU/Feature table in TSV format
  -b, --biom <biom>           OTU/Feature table in BIOM (HDF5) format
  -o, --output <output>       Output distance matrix in TSV format [default: unifrac.tsv]
  -s, --sketch <sketch-size>  Sketch size for Weighted MinHash (DartMinHash or ERS) [default: 1024]
  -m, --method <method>       Sketching method: dmh (DartMinHash) or ers (Efficient Rejection Sampling) [default: dmh] [possible values: dmh, ers]
      --length <seq-length>   Per-hash independent random sequence length L for ERS [default: 1024]
      --seed <seed>           Random seed for reproducibility [default: 1337]
  -h, --help                  Print help
  -V, --version               Print version

```

## Benchmark

We use Striped UniFrac as the ground truth. A pure Rust implementaion can be [here](https://github.com/jianshu93/unifrac_bp).

## Acknowledgements
We want to thank Otmar Ertl and Xiaoyun Li for their helpful comments on DartMinHash and Efficient Rejection Sampling, respectively. We want to thank Sherlyn Weng for helping with DartUniFrac logo design.

## References

Paper to come