"""
Subcommands:
  generate  -- create a synthetic instance and save it to JSON
  solve     -- solve a saved instance with gurobi, lru, or random
  verify    -- check that a solution satisfies all constraints
  plot      -- generate charts from a solution
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from src import data_generator, instance_io, model_gurobi, visualize
from src.baselines.lru import LRUPolicy
from src.baselines.random_evict import RandomPolicy
from src.baselines.simulator import simulate
from src.solution import solution_to_dict, solution_from_dict, verify

def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# subcommand handlers
def cmd_generate(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    seed = args.seed if args.seed is not None else cfg["instance"]["seed"]
    inst = data_generator.generate(cfg, seed=seed)
    out = args.out or "results/instance.json"
    instance_io.save(inst, out)
    print(f"Instance saved to {out}")
    print(f"  adapters={len(inst.adapters)}, gpus={len(inst.gpus)}, "
          f"requests={len(inst.requests)}, time_slots={inst.n_time_slots}")

def cmd_solve(args: argparse.Namespace) -> None:
    inst = instance_io.load(args.instance)
    cfg = _load_cfg(args.config)

    if args.solver == "gurobi":
        sol = model_gurobi.build_and_solve(inst, cfg["solver"])
    elif args.solver == "lru":
        sol = simulate(inst, LRUPolicy())
    elif args.solver == "random":
        sol = simulate(inst, RandomPolicy(np.random.default_rng(0)))
    else:
        print(f"Unknown solver: {args.solver}", file=sys.stderr)
        sys.exit(1)

    out = args.out or f"results/solution_{args.solver}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(solution_to_dict(sol), f, indent=2)
    print(f"Solution saved to {out}  [status={sol.status}, obj={sol.objective:.4f}]")

def cmd_verify(args: argparse.Namespace) -> None:
    inst = instance_io.load(args.instance)
    with open(args.solution) as f:
        sol = solution_from_dict(json.load(f))

    violations = verify(sol, inst)
    if violations:
        print(f"FAILED — {len(violations)} violation(s):")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("OK — no violations found.")

def cmd_plot(args: argparse.Namespace) -> None:
    inst = instance_io.load(args.instance)
    with open(args.solution) as f:
        sol = solution_from_dict(json.load(f))

    out_dir = Path(args.out or "results/charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    visualize.plot_adapter_schedule(sol, inst, out_dir / "adapter_schedule.png")
    visualize.plot_gpu_utilization(sol, inst, out_dir / "gpu_utilization.png")
    visualize.plot_priority_delay(sol, inst, out_dir / "priority_delay.png")
    print(f"Charts saved to {out_dir}/")

# argument parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="LoRA adapter swap optimizer CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Create a synthetic problem instance")
    p_gen.add_argument("--config", default="config/default.yaml")
    p_gen.add_argument("--out", help="Output JSON path (default: results/instance.json)")
    p_gen.add_argument("--seed", type=int, default=None)

    p_sol = sub.add_parser("solve", help="Solve a problem instance")
    p_sol.add_argument("--instance", required=True, help="Path to instance JSON")
    p_sol.add_argument("--solver", choices=["gurobi", "lru", "random"], default="gurobi")
    p_sol.add_argument("--config", default="config/default.yaml")
    p_sol.add_argument("--out", help="Output JSON path")

    p_ver = sub.add_parser("verify", help="Check a solution for constraint violations")
    p_ver.add_argument("--instance", required=True)
    p_ver.add_argument("--solution", required=True)

    p_plt = sub.add_parser("plot", help="Generate charts from a solution")
    p_plt.add_argument("--instance", required=True)
    p_plt.add_argument("--solution", required=True)
    p_plt.add_argument("--out", help="Output directory (default: results/charts)")

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "generate": cmd_generate,
        "solve": cmd_solve,
        "verify": cmd_verify,
        "plot": cmd_plot,
    }
    dispatch[args.command](args)

if __name__ == "__main__":
    main()