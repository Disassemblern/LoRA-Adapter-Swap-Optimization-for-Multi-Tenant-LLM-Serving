"""
Parameter sweep: vary instance size, collect cost + timing data for Gurobi and SA.

Usage (from project root):
    python experiments/run_sweep.py
    python experiments/run_sweep.py --out results/sweep/sweep.csv
"""
from __future__ import annotations
import argparse
import copy
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import data_generator, model_gurobi
from src.baselines.simulated_annealing import SimulatedAnnealing
from src import metrics


SWEEP_GRID = {
    "n_requests": [50, 100, 150, 200],
    "n_adapters": [8, 10, 12, 15],
}
SEEDS = [42, 123, 7]

COLUMNS = [
    "seed", "sweep_param", "sweep_value",
    "solver", "total", "latency", "swap", "priority",
    "mip_gap", "wall_time",
]


def run_cell(cfg: dict, seed: int, solver_name: str) -> dict:
    inst = data_generator.generate(cfg, seed=seed)

    if solver_name == "gurobi":
        sol = model_gurobi.build_and_solve(inst, cfg["solver"])
        gap = sol.mip_gap
        wall = sol.wall_time_sec
    else:
        sol = SimulatedAnnealing(cfg.get("simulated_annealing", {})).solve(inst)
        gap = 0.0
        wall = sol.wall_time_sec

    m = metrics.decompose(sol, inst)
    return {**m, "mip_gap": gap, "wall_time": wall}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--out", default="results/sweep/sweep.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_runs = sum(len(v) for v in SWEEP_GRID.values()) * len(SEEDS) * 2
    done = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for param, values in SWEEP_GRID.items():
            for value in values:
                cfg = copy.deepcopy(base_cfg)
                cfg["instance"][param] = value

                for seed in SEEDS:
                    for solver in ["gurobi", "simulated_annealing"]:
                        done += 1
                        print(f"[{done}/{total_runs}] {param}={value} seed={seed} solver={solver}")
                        try:
                            result = run_cell(cfg, seed, solver)
                            writer.writerow({
                                "seed": seed,
                                "sweep_param": param,
                                "sweep_value": value,
                                "solver": solver,
                                **result,
                            })
                            f.flush()
                        except Exception as e:
                            print(f"  ERROR: {e}")

    print(f"\nSweep complete. Results saved to {out_path}")


if __name__ == "__main__":
    main()
