from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.data_types import Instance
from src.solution import Solution
from src import metrics


def plot_adapter_schedule(sol: Solution, inst: Instance, path: str | Path) -> None:
    """
    x=time slot, one row per (GPU, adapter).
    A filled cell means the adapter is loaded on that GPU at that time.
    A red border on a cell means a swap-in happened at that slot.
    """
    n_gpus = len(inst.gpus)
    n_adapters = len(inst.adapters)
    n_slots = inst.n_time_slots

    colors = plt.cm.tab10(np.linspace(0, 1, n_adapters))

    fig, axes = plt.subplots(n_gpus, 1, figsize=(max(8, n_slots), n_gpus * 2.5),
                             sharex=True)
    if n_gpus == 1:
        axes = [axes]

    for ax, g in zip(axes, inst.gpus):
        ax.set_title(f"GPU {g.id}  (budget={g.adapter_budget_gb:.1f} GB, tp={g.max_throughput})")
        ax.set_ylabel("Adapter")
        ax.set_yticks(range(n_adapters))
        ax.set_yticklabels([f"A{a.id} ({a.memory_gb:.1f}GB)" for a in inst.adapters])
        ax.set_xlim(-0.5, n_slots - 0.5)

        for row, a in enumerate(inst.adapters):
            for t in range(n_slots):
                loaded = sol.adapter_schedule.get((a.id, g.id, t), False)
                swapped = sol.swaps.get((a.id, g.id, t), False)
                if loaded:
                    rect = mpatches.FancyBboxPatch(
                        (t - 0.45, row - 0.4), 0.9, 0.8,
                        boxstyle="round,pad=0.05",
                        facecolor=colors[row],
                        edgecolor="red" if swapped else "white",
                        linewidth=2 if swapped else 0.5,
                        alpha=0.85,
                    )
                    ax.add_patch(rect)

        ax.set_xticks(range(n_slots))

    axes[-1].set_xlabel("Time slot")
    fig.suptitle("Adapter Loading Schedule  (red border = swap-in)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_gpu_utilization(sol: Solution, inst: Instance, path: str | Path) -> None:
    """
    Stacked bar chart: x=time slot, y=GB of adapters loaded per GPU.
    Each adapter has its own color. The dashed line shows the VRAM budget.
    """
    n_slots = inst.n_time_slots
    colors = plt.cm.tab10(np.linspace(0, 1, len(inst.adapters)))

    fig, axes = plt.subplots(len(inst.gpus), 1,
                             figsize=(max(8, n_slots), len(inst.gpus) * 3),
                             sharex=True)
    if len(inst.gpus) == 1:
        axes = [axes]

    for ax, g in zip(axes, inst.gpus):
        bottoms = np.zeros(n_slots)
        for i, a in enumerate(inst.adapters):
            heights = np.array([
                a.memory_gb if sol.adapter_schedule.get((a.id, g.id, t), False) else 0.0
                for t in range(n_slots)
            ])
            ax.bar(range(n_slots), heights, bottom=bottoms,
                   color=colors[i], label=f"A{a.id}", alpha=0.85)
            bottoms += heights

        ax.axhline(g.adapter_budget_gb, color="red", linestyle="--",
                   linewidth=1.2, label=f"budget ({g.adapter_budget_gb:.1f} GB)")
        ax.set_title(f"GPU {g.id} VRAM usage")
        ax.set_ylabel("GB")
        ax.legend(loc="upper right", fontsize=7, ncol=3)

    axes[-1].set_xlabel("Time slot")
    axes[-1].set_xticks(range(n_slots))
    fig.suptitle("GPU VRAM Utilization per Time Slot", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cost_breakdown(
    sols_by_name: dict[str, Solution], inst: Instance, path: str | Path
) -> None:
    """
    Grouped bar chart comparing latency / swap / priority cost across solvers.
    """
    names = list(sols_by_name.keys())
    components = ["latency", "swap", "priority"]
    data = {c: [] for c in components}

    for name in names:
        m = metrics.decompose(sols_by_name[name], inst)
        for c in components:
            data[c].append(m[c])

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))

    bar_colors = ["#4878CF", "#6ACC65", "#D65F5F"]
    for i, (comp, color) in enumerate(zip(components, bar_colors)):
        bars = ax.bar(x + i * width, data[comp], width, label=comp, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(names)
    ax.set_ylabel("Cost")
    ax.set_title("Cost Breakdown by Solver")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_priority_delay(sol: Solution, inst: Instance, path: str | Path) -> None:
    """
    Box plot of per-request latency grouped by priority level.
    Shows whether the optimizer genuinely routes high-priority requests faster.
    """
    latencies_by_priority: dict[int, list[float]] = {1: [], 2: [], 3: []}

    for req in inst.requests:
        if req.id in sol.assignments:
            j = sol.assignments[req.id]
            latencies_by_priority[req.priority].append(req.latency_per_gpu[j])

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [latencies_by_priority[p] for p in [1, 2, 3]]
    labels = ["Low (1)", "Medium (2)", "High (3)"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    box_colors = ["#D65F5F", "#6ACC65", "#4878CF"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xlabel("Request Priority")
    ax.set_ylabel("Latency")
    ax.set_title("Latency Distribution by Priority")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
