"""
Run Gurobi and Genetic Algorithm on one instance and compare costs.

Usage (from project root):
    python experiments/run_single.py
    python experiments/run_single.py --config config/default.yaml --seed 42
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import data_generator, model_gurobi, visualize
from src.baselines.genetic import GeneticAlgorithm
from src import metrics
from src.solution import verify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="results", help="Output directory for charts")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["instance"]["seed"]
    print(f"Generating instance (seed={seed})...")
    inst = data_generator.generate(cfg, seed=seed)
    print(f"  {len(inst.adapters)} adapters, {len(inst.gpus)} GPUs, "
          f"{len(inst.requests)} requests, {inst.n_time_slots} time slots")

    # solve
    print("\nSolving with Gurobi...")
    sol_gurobi = model_gurobi.build_and_solve(inst, cfg["solver"])
    print(f"  status={sol_gurobi.status}, obj={sol_gurobi.objective:.4f}, "
          f"gap={sol_gurobi.mip_gap*100:.2f}%, time={sol_gurobi.wall_time_sec:.2f}s")

    print("Solving with Genetic Algorithm...")
    sol_ga = GeneticAlgorithm(cfg.get("genetic", {})).solve(inst)
    print(f"  status={sol_ga.status}, obj={sol_ga.objective:.4f}, "
          f"time={sol_ga.wall_time_sec:.2f}s")

    # verify
    print("\nVerifying solutions...")
    all_ok = True
    for name, sol in [("gurobi", sol_gurobi), ("genetic", sol_ga)]:
        violations = verify(sol, inst)
        if violations:
            print(f"  [{name}] FAILED — {len(violations)} violation(s):")
            for v in violations:
                print(f"    - {v}")
            all_ok = False
        else:
            print(f"  [{name}] OK")

    if not all_ok:
        print("\nAborting: one or more solutions failed verification.")
        sys.exit(1)

    # cost comparison
    print("\nCost breakdown:")
    print(f"  {'solver':<10} {'total':>8} {'latency':>8} {'swap':>8} {'priority':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    sols = {"gurobi": sol_gurobi, "genetic": sol_ga}
    for name, sol in sols.items():
        m = metrics.decompose(sol, inst)
        print(f"  {name:<10} {m['total']:>8.3f} {m['latency']:>8.3f} "
              f"{m['swap']:>8.3f} {m['priority']:>8.3f}")

    gurobi_total = metrics.decompose(sol_gurobi, inst)["total"]
    ga_total = metrics.decompose(sol_ga, inst)["total"]
    print(f"\n  Gurobi vs GA: {(ga_total - gurobi_total) / ga_total * 100:+.1f}% cost reduction by Gurobi")

    # charts
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving charts to {out_dir}/...")
    visualize.plot_cost_breakdown(sols, inst, out_dir / "cost_breakdown.png")
    visualize.plot_adapter_schedule(sol_gurobi, inst, out_dir / "schedule_gurobi.png")
    visualize.plot_gpu_utilization(sol_gurobi, inst, out_dir / "utilization_gurobi.png")
    visualize.plot_priority_delay(sol_gurobi, inst, out_dir / "priority_delay_gurobi.png")
    print("Done.")


if __name__ == "__main__":
    main()
