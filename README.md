# LoRA Adapter Swap Optimization for Multi-Tenant LLM Serving

Optimal LoRA adapter loading and request routing for multi-tenant LLM serving.
Formulated as a Mixed-Integer Program (MIP) and solved with Gurobi, with a
Genetic Algorithm metaheuristic as a comparison baseline.

---

## Problem

In production LLM serving, a single base model is shared across many tasks via
LoRA adapters. Each adapter must be loaded into GPU VRAM before it can serve a
request. VRAM is limited, so adapters are swapped in and out dynamically —
and every swap costs time.

The goal is to find the joint **adapter loading schedule** and **request routing**
that minimizes total cost across three competing objectives.

---

## Mathematical Formulation

### Decision Variables

| Variable | Type | Description |
|---|---|---|
| `x[i, j]` | Binary | 1 if request `i` is assigned to GPU `j` |
| `y[a, j, t]` | Binary | 1 if adapter `a` is loaded on GPU `j` at time slot `t` |
| `s[a, j, t]` | Binary | 1 if adapter `a` is swapped IN on GPU `j` at time slot `t` |
| `delay[i]` | Continuous ≥ 0 | Effective latency experienced by request `i` |

### Objective Function

Minimize the weighted sum of three cost terms:

```
minimize:
    α · Σ_{i,j}   latency[i,j]  · x[i,j]        (total serving latency)
  + β · Σ_{a,j,t} swap_cost[a]  · s[a,j,t]       (total swap overhead)
  + γ · Σ_i       priority[i]   · delay[i]        (weighted priority delay)
```

### Weights (tunable in `config/default.yaml`)

| Symbol | Config key | Default | Meaning |
|---|---|---|---|
| `α` (alpha) | `objective.alpha` | `1.0` | How much raw serving latency matters |
| `β` (beta) | `objective.beta` | `2.0` | How much adapter swap cost matters — set higher than alpha because swaps are a systemic overhead that compounds across all requests |
| `γ` (gamma) | `objective.gamma` | `0.5` | How much priority-weighted delay matters — ensures high-priority requests are preferentially routed to faster GPUs |

### Constraints

**1. Assignment** — every request goes to exactly one GPU:
```
Σ_j x[i,j] = 1    for all i
```

**2. Adapter availability** — a request can only go to a GPU that has its adapter loaded at arrival time:
```
x[i,j] ≤ y[adapter(i), j, arrival(i)]    for all i, j
```

**3. Memory capacity** — loaded adapters cannot exceed a GPU's free VRAM (total VRAM minus base model):
```
Σ_a memory[a] · y[a,j,t] ≤ VRAM[j] - base_model_gb    for all j, t
```

**4. Swap detection** — `s` fires whenever an adapter transitions from unloaded to loaded:
```
s[a,j,t] ≥ y[a,j,t] - y[a,j,t-1]    for all a, j, t ≥ 1
s[a,j,0] ≥ y[a,j,0]                  (GPUs start empty at t=0)
```

**5. Throughput** — each GPU handles at most `max_throughput` requests per time slot:
```
Σ_{i: arrival(i)=t} x[i,j] ≤ max_throughput[j]    for all j, t
```

**6. Delay linkage** — `delay[i]` is at least as large as the latency of whichever GPU handles request `i` (this is what makes the γ priority term meaningful):
```
delay[i] ≥ latency[i,j] · x[i,j]    for all i, j
```

---

## Solvers

### Gurobi (exact)
Solves the MIP to proven optimality (or within a configurable MIP gap).
Requires a Gurobi license — academic and WLS licenses work.

### Genetic Algorithm (metaheuristic)
A population-based search used to benchmark how close a license-free heuristic
can get to the Gurobi optimum.

- **Chromosome**: integer array of length `n_requests` where `chrom[i] = gpu_id`
- **Crossover**: uniform (gene-by-gene), rate `0.85`
- **Mutation**: per-gene random GPU reassignment, rate `0.02`
- **Selection**: tournament (size 4)
- **Elitism**: top 2 chromosomes carried unchanged each generation
- **Repair**: throughput violations fixed deterministically after each crossover/mutation
- **Adapter schedule**: derived greedily from assignments (largest-evict when VRAM full)

---

## Project Structure

```
src/
  data_types.py        — Adapter, GPU, Request, Instance dataclasses
  data_generator.py    — synthetic instance generation
  instance_io.py       — save/load instances as JSON
  model_gurobi.py      — MIP formulation and Gurobi solve
  solution.py          — Solution dataclass, serialization, constraint verifier
  metrics.py           — cost decomposition (latency / swap / priority)
  visualize.py         — matplotlib charts
  cli.py               — CLI entry point
  baselines/
    genetic.py         — Genetic Algorithm solver

config/
  default.yaml         — all tunable parameters

experiments/
  run_single.py        — solve one instance with both solvers, print comparison
  run_sweep.py         — parameter sweep, writes results/sweep.csv

notebooks/
  analysis.ipynb       — load sweep CSV, plot gap%, wall time, cost breakdown

tests/
  test_data_generator.py
  test_model_gurobi.py
  test_verifier.py
  test_baselines.py
```

---

## Configuration Reference (`config/default.yaml`)

### Instance

| Key | Default | Description |
|---|---|---|
| `n_adapters` | `10` | Number of distinct LoRA adapters |
| `n_gpus` | `3` | Number of GPUs in the cluster |
| `n_requests` | `100` | Total incoming requests to schedule |
| `n_time_slots` | `15` | Number of discrete time steps |
| `adapter_memory_gb.min/max` | `0.5 / 4.0` | Adapter VRAM footprint range (GB) |
| `vram_gb.min/max` | `24 / 80` | GPU total VRAM range (GB) |
| `base_model_gb` | `14.0` | VRAM reserved for the base model on every GPU |
| `arrival` | `zipf` | Request distribution — `zipf` concentrates demand on a few adapters |
| `zipf_alpha` | `1.2` | Zipf skew — higher means more requests hit the top adapter |
| `seed` | `42` | Random seed for reproducibility |

### Solver (Gurobi)

| Key | Default | Description |
|---|---|---|
| `time_limit_sec` | `300` | Stop after this many seconds even if not optimal |
| `mip_gap` | `0.01` | Accept solutions within 1% of proven optimum |
| `threads` | `0` | CPU threads — `0` lets Gurobi choose automatically |
| `log_to_console` | `true` | Print Gurobi solving progress |

### Genetic Algorithm

| Key | Default | Description |
|---|---|---|
| `population_size` | `60` | Number of chromosomes per generation |
| `n_generations` | `200` | How many generations to evolve |
| `crossover_rate` | `0.85` | Probability two parents exchange genes |
| `mutation_rate` | `0.02` | Per-gene probability of random GPU reassignment |
| `tournament_size` | `4` | Candidates per selection tournament |
| `elite_count` | `2` | Top individuals copied unchanged each generation |
| `seed` | `0` | Random seed for reproducibility |

---

## Usage

```bash
# Generate a synthetic instance
python -m src.cli generate --config config/default.yaml --out results/instance.json

# Solve with Gurobi
python -m src.cli solve --instance results/instance.json --solver gurobi --out results/solution_gurobi.json

# Solve with Genetic Algorithm
python -m src.cli solve --instance results/instance.json --solver genetic --out results/solution_genetic.json

# Verify a solution satisfies all constraints
python -m src.cli verify --instance results/instance.json --solution results/solution_gurobi.json

# Generate charts
python -m src.cli plot --instance results/instance.json --solution results/solution_gurobi.json --out results/charts

# Run both solvers and compare in one shot
python experiments/run_single.py

# Parameter sweep (writes results/sweep.csv)
python experiments/run_sweep.py

# Run tests
pytest -q
```

---

## Requirements

```
gurobipy >= 11.0
numpy >= 1.26
pandas >= 2.1
matplotlib >= 3.8
pyyaml >= 6.0
pytest >= 8.0
```
