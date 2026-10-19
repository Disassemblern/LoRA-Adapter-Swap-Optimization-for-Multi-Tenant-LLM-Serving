# LoRA Adapter Swap Optimization for Multi-Tenant LLM Serving

Optimal LoRA adapter loading and request routing for multi-tenant LLM serving.
Formulated as a Mixed-Integer Program (MIP) and solved with Gurobi, with a
Simulated Annealing metaheuristic as a comparison baseline.

---

## Problem

In production LLM serving, a single base model is shared across many tasks via
LoRA adapters. Each adapter must be loaded into GPU VRAM before it can serve a
request. VRAM is limited, so adapters are swapped in and out dynamically
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

### Simulated Annealing (metaheuristic)
A single-solution local search used to benchmark how close a license-free heuristic
can get to the Gurobi optimum.

- **Solution encoding**: integer array of length `n_requests` where `solution[i] = gpu_id`
- **Neighbourhood move**: reassign one randomly chosen request to a different GPU
- **Acceptance**: always accept improvements; accept worse solutions with probability `exp(-delta / T)`
- **Cooling schedule**: geometric decay — `T(k) = T_start * cooling_rate^k`
- **Repair**: throughput violations fixed deterministically after every move
- **Adapter schedule**: derived greedily from assignments (largest-evict when VRAM full)

### Results (default config, seed 42)

| Solver | Total | Latency | Swap | Priority | Time |
|---|---|---|---|---|---|
| Gurobi | 287.889 | 109.393 | 31.120 | 232.510 | 0.07 s (optimal, gap < 1%) |
| Simulated Annealing | 308.091 | 110.802 | 39.612 | 236.129 | 0.67 s |

Across the parameter sweep (`experiments/run_sweep.py`), Gurobi's cost is
5-15% lower than SA's (mean about 9%), and Gurobi solves every instance in
under 0.5 s.

> **Correction.** An earlier version reported SA at 1320.7. That figure came
> from a bug in SA's schedule builder: every loaded adapter was counted as a
> swap in every time slot. Swaps are now derived from the adapter schedule
> for every solver (`solution.swaps_from_schedule`), and the verifier rejects
> schedules whose loads are not reported as swaps.

---

## Project Structure

```
src/
  data_types.py        — Adapter, GPU, Request, Instance dataclasses
  data_generator.py    — synthetic instance generation
  instance_io.py       — save/load instances as JSON
  model_gurobi.py      — MIP formulation and Gurobi solve
  solution.py          — Solution dataclass, serialization, constraint verifier
  metrics.py           — cost decomposition (latency / swap / priority; swaps derived from the schedule)
  visualize.py         — matplotlib charts
  cli.py               — CLI entry point
  baselines/
    simulated_annealing.py — Simulated Annealing solver

config/
  default.yaml         — all tunable parameters

experiments/
  run_single.py        — solve one instance with both solvers, print comparison
  run_sweep.py         — parameter sweep, writes results/sweep/sweep.csv
  analysis.ipynb       — load sweep CSV, plot gap%, wall time, cost breakdown

docs/
  report.tex           — project report (LaTeX source)

tests/
  test_data_generator.py
  test_model_gurobi.py
  test_verifier.py
  test_baselines.py
  test_metrics.py

results/               — generated, git-ignored; one subfolder per script
  cli/  single/  sweep/
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

### Simulated Annealing

| Key | Default | Description |
|---|---|---|
| `t_start` | `500.0` | Initial temperature |
| `t_min` | `0.1` | Stop cooling below this temperature |
| `cooling_rate` | `0.995` | Geometric decay factor applied each iteration |
| `n_iterations` | `10000` | Maximum number of iterations |
| `seed` | `0` | Random seed for reproducibility |

---

## Usage

```bash
# Generate a synthetic instance           -> results/cli/instance.json
python -m src.cli generate

# Solve with Gurobi or Simulated Annealing -> results/cli/solution_<solver>.json
python -m src.cli solve --instance results/cli/instance.json --solver gurobi
python -m src.cli solve --instance results/cli/instance.json --solver simulated_annealing

# Verify a solution satisfies all constraints
python -m src.cli verify --instance results/cli/instance.json --solution results/cli/solution_gurobi.json

# Generate charts                          -> results/cli/charts/
python -m src.cli plot --instance results/cli/instance.json --solution results/cli/solution_gurobi.json

# Run both solvers and compare in one shot -> results/single/
python experiments/run_single.py

# Parameter sweep                          -> results/sweep/sweep.csv
python experiments/run_sweep.py

# Run tests
pytest -q
```

---

## Requirements

Install with `pip install -e ".[dev]"` (dependencies are listed in `pyproject.toml`).
The pip build of gurobipy includes a size-limited license (2000 variables and
constraints), which covers the default instance and all tests.
