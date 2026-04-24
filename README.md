"""# Fugaku Experiments Folder

## 📂 Folder Structure

```
Fugaku_experiments/
├── Allgather/
│   ├── all_gather_radix_batch.cpp
│   └── main.cpp
├── Allreduce/
│   ├── all_reduce_radix_batch.cpp
│   └── main.cpp
├── Reduce-scatter/
│   ├── reduce_scatter_radix_batch.cpp
│   └── main.cpp
└── README.md
```

Each `main.cpp` follows a consistent structure and benchmarking flow:
1. Initializes MPI and parses runtime parameters.
2. Iterates over different message sizes (`count = base << i`).
3. Benchmarks radix- and batch-parameterized algorithm variants.
4. Records results to a CSV file (`results<num_nodes>.csv`).

---

## ⚙️ Compilation

All programs can be compiled using `mpic++`:

```bash
mpic++ -O3 -std=c++17 -o allgather main.cpp all_gather_radix_batch.cpp
mpic++ -O3 -std=c++17 -o allreduce main.cpp all_reduce_radix_batch.cpp
mpic++ -O3 -std=c++17 -o reduce_scatter main.cpp reduce_scatter_radix_batch.cpp
```

---

## 🚀 Usage

Each executable accepts the same set of parameters:

```bash
mpirun -np <nprocs> ./program <n_iter> [--overwrite] [b=<value>] [base=<value>] [num_nodes=<value>] [radix_increment=<value>]
```

### **Arguments**

| Parameter | Description | Default |
|------------|-------------|----------|
| `<n_iter>` | Number of iterations (required) | — |
| `--overwrite` | Overwrites existing CSV file | false |
| `b=<value>` | Batch size parameter (upper limit for radix loop) | 16 |
| `base=<value>` | Base message size multiplier | 1 |
| `num_nodes=<value>` | Number of nodes in the system (used for CSV naming) | 1 |
| `radix_increment=<value>` | Step size for increasing radix `r` | 1 |

Example:
```bash
mpirun -np 16 ./reduce_scatter 8 --overwrite b=32 base=4 num_nodes=4 radix_increment=2
```

---

## 🧠 Output

Each run produces a CSV file named:

```
results<num_nodes>.csv
```

The CSV contains one row per experiment configuration:

| algorithm_name | k | b | nprocs | send_count | time | is_correct |
|-----------------|---|---|--------|-------------|------|-------------|
| reduce_scatter_radix_batch | 4 | 16 | 32 | 4096 | 1.23e-05 | 1 |
| reduce_scatter_standard | — | — | 32 | 4096 | 2.45e-05 | 1 |

- **`time`** is the measured execution latency in seconds.  
- **`is_correct`** indicates whether the result was validated against the MPI reference collective.

---

## 🧩 Implementation Notes

- Each collective (AllGather, AllReduce, ReduceScatter) defines:
  - A *parameterized radix/batch variant* (`*_radix_batch.cpp`)
  - A *standard MPI baseline* for comparison.
- The test harness in `main.cpp` automatically sweeps message sizes and radix values.
- Designed for scaling studies on HPC clusters (e.g., Fugaku, Polaris, Aurora).

---

## 📜 Example Run

```bash
# Example: Run AllReduce experiment on 8 processes
mpirun -np 8 ./allreduce 6 --overwrite b=16 base=2 num_nodes=8 radix_increment=2
```

Output:
```
Parsed parameters:
  n_iter          = 6 
  overwrite       = true
  b               = 16
  base            = 2
  num_nodes       = 8
  radix_increment = 2
```

The resulting file `results8.csv` will contain performance data for all tested configurations.

"""