"""
Aggregated (flow) MILP for LoRA adapter placement and routing.

Decisions per adapter a, GPU j, time slot t (slot length slot_s):
  y[a,j,t] binary  adapter in the GPU LoRA pool   (SGLang --max-loras-per-batch)
  z[a,j,t] binary  adapter in CPU memory          (SGLang --max-loaded-loras)
  u[a,j,t] >= 0    load into the GPU pool    (u >= y_t - y_{t-1})
  p[a,j,t] >= 0    insert from disk into CPU (p >= z_t - z_{t-1})
  f[a,j,t] >= 0    requests of a served on j in t
  b[a,t]   >= 0    requests of a still waiting after slot t

Constraints:
  (1) Demand balance:  b[a,t-1] + demand[a,t] = sum_j f[a,j,t] + b[a,t]
  (2) Residency:       f[a,j,t] <= (K_j / w_a) * y[a,j,t]
  (3) Token capacity:  sum_a w_a * f[a,j,t] <= K_j
  (4) GPU pool:        sum_a y[a,j,t] <= S_j
  (5) CPU memory:      sum_a z[a,j,t] <= C_j,  z >= y
  (6) Load detection:  u >= y_t - y_{t-1},  p >= z_t - z_{t-1}  (t=-1 is the given start state)

Objective (request-seconds, SLO-weighted):
  sum_{a,t} w_slo[a] * slot_s * b[a,t]                                  backlog waiting
  + sum_{a,j,t} c[a,t] * (cpu_s * u + (disk_s - cpu_s) * p)             waiting caused by loads
  with c[a,t] = w_slo[a] * (1 + demand[a,t] * disk_s / slot_s)

u and p need no integrality: y and z are binary and u, p have positive cost,
so the optimum sets them to exactly max(0, y_t - y_{t-1}) and max(0, z_t - z_{t-1}).
"""
from __future__ import annotations
import time

import gurobipy as gp
from gurobipy import GRB

from src.cluster.types import FlowInstance, FlowPlan
from src.model_gurobi import apply_solver_cfg, status_name


def solve(inst: FlowInstance, solver_cfg: dict) -> FlowPlan:
    m = gp.Model("lora_flow")
    apply_solver_cfg(m, solver_cfg)

    keys = [(a, j, t)
            for a in range(len(inst.adapters))
            for j in range(len(inst.gpus))
            for t in range(inst.horizon)]
    y = m.addVars(keys, vtype=GRB.BINARY, name="y")
    z = m.addVars(keys, vtype=GRB.BINARY, name="z")
    u = m.addVars(keys, lb=0.0, name="u")
    p = m.addVars(keys, lb=0.0, name="p")
    f = m.addVars(keys, lb=0.0, name="f")
    b = m.addVars([(a, t) for a in range(len(inst.adapters)) for t in range(inst.horizon)],
                  lb=0.0, name="b")

    _add_demand_balance(m, inst, f, b)
    _add_gpu_constraints(m, inst, y, z, f)
    _add_load_detection(m, inst, y, z, u, p)
    _set_objective(m, inst, u, p, b)

    t0 = time.time()
    m.optimize()
    wall = time.time() - t0
    return _extract(m, inst, y, z, u, p, f, b, wall)


def _initial_backlog(inst: FlowInstance, a: int) -> float:
    return inst.backlog[a] if inst.backlog else 0.0


def _add_demand_balance(m, inst, f, b) -> None:
    # (1) carried-over + new demand is either served this slot or keeps waiting
    for a in range(len(inst.adapters)):
        for t in range(inst.horizon):
            carried = b[a, t - 1] if t > 0 else _initial_backlog(inst, a)
            m.addConstr(
                carried + inst.demand[a][t]
                == gp.quicksum(f[a, j, t] for j in range(len(inst.gpus))) + b[a, t],
                name=f"bal_{a}_{t}",
            )


def _add_gpu_constraints(m, inst, y, z, f) -> None:
    n_a = len(inst.adapters)
    for j, g in enumerate(inst.gpus):
        for t in range(inst.horizon):
            for a, ad in enumerate(inst.adapters):
                # (2) only adapters in the GPU pool can serve
                m.addConstr(f[a, j, t] <= (g.tokens_per_slot / ad.tokens_per_request) * y[a, j, t],
                            name=f"res_{a}_{j}_{t}")
                # (5) the GPU pool is a subset of CPU memory
                m.addConstr(z[a, j, t] >= y[a, j, t], name=f"tier_{a}_{j}_{t}")
            # (3) token throughput per slot
            m.addConstr(gp.quicksum(ad.tokens_per_request * f[a, j, t]
                                    for a, ad in enumerate(inst.adapters)) <= g.tokens_per_slot,
                        name=f"cap_{j}_{t}")
            # (4) GPU LoRA pool slots, (5) CPU memory slots
            m.addConstr(gp.quicksum(y[a, j, t] for a in range(n_a)) <= g.lora_slots,
                        name=f"gpu_{j}_{t}")
            m.addConstr(gp.quicksum(z[a, j, t] for a in range(n_a)) <= g.cpu_slots,
                        name=f"cpu_{j}_{t}")


def _add_load_detection(m, inst, y, z, u, p) -> None:
    # (6) a load happens when an adapter enters a tier; t=-1 is the start state
    for a in range(len(inst.adapters)):
        for j in range(len(inst.gpus)):
            for t in range(inst.horizon):
                y_prev = y[a, j, t - 1] if t > 0 else float((a, j) in inst.gpu_loaded)
                z_prev = z[a, j, t - 1] if t > 0 else float((a, j) in inst.cpu_loaded)
                m.addConstr(u[a, j, t] >= y[a, j, t] - y_prev, name=f"u_{a}_{j}_{t}")
                m.addConstr(p[a, j, t] >= z[a, j, t] - z_prev, name=f"p_{a}_{j}_{t}")


def _set_objective(m, inst, u, p, b) -> None:
    cpu_s = inst.load.cpu_to_gpu_s
    disk_extra_s = inst.load.disk_to_gpu_s - cpu_s
    backlog_wait = gp.quicksum(
        ad.slo_weight * inst.slot_s * b[a, t]
        for a, ad in enumerate(inst.adapters)
        for t in range(inst.horizon)
    )
    # ponytail: first-order estimate of who waits for a load (the triggering
    # request plus arrivals during a disk load); recalibrate after the E5 replay
    load_wait = gp.quicksum(
        ad.slo_weight * (1 + inst.demand[a][t] * inst.load.disk_to_gpu_s / inst.slot_s)
        * (cpu_s * u[a, j, t] + disk_extra_s * p[a, j, t])
        for a, ad in enumerate(inst.adapters)
        for j in range(len(inst.gpus))
        for t in range(inst.horizon)
    )
    m.setObjective(backlog_wait + load_wait, GRB.MINIMIZE)


def _extract(m, inst, y, z, u, p, f, b, wall: float) -> FlowPlan:
    status = status_name(m)
    if m.SolCount == 0:
        raise RuntimeError(f"Gurobi found no feasible plan (status={status}).")
    return FlowPlan(
        gpu_loaded=frozenset(k for k, v in y.items() if v.X > 0.5),
        cpu_loaded=frozenset(k for k, v in z.items() if v.X > 0.5),
        served={k: v.X for k, v in f.items() if v.X > 1e-6},
        backlog={k: v.X for k, v in b.items()},
        gpu_loads=round(sum(v.X for v in u.values())),
        cpu_inserts=round(sum(v.X for v in p.values())),
        objective_s=m.ObjVal,
        status=status,
        mip_gap=m.MIPGap,
        wall_time_s=wall,
        stats={"n_vars": m.NumVars, "n_constrs": m.NumConstrs},
    )
