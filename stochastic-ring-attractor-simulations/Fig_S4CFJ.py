import os
import sys
import json
import math
import shutil
import random
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm  # for progress bar

# Optional/leftover imports you might use elsewhere
import matplotlib.pyplot as plt  
from matplotlib.colors import Normalize  

# --- your project imports ---
from core.spinSystem import *            
from core.perceptionSystem import *      
from core.environment import *           
from core.agent import *                 
from core.simulation import *            
from core.experiment_utils import code_fingerprint, H5ExperimentWriter

# -----------------------------


# ==========================
# Utilities
# ==========================


def default_base_params():
    return {
        "goals": [],  # will be set per angle
        "num_groups": 32,
        "num_spins_per_group": 10,
        "T": 0.4,
        "suppression_scaling": 5,
        "intrinsic_stimulation": 0.0,
        "J": 18,
        "nu": 0.5,
        "dynamics": "glauber",
        "dt": 1,
        "tau": 160,
        "step_size": 0.01,
        "initial_position": (0, 0),
        "initial_orientation": 0,
        "stop_on_goal": False,
        "time_delay": 0,
        "egocentric": False,
        "prerun": True,
        "prerun_config": {"distribution": "von_mises", "peak_angle": 0},
    }


def parse_csv_floats(val):
    if not val:
        return None
    out = []
    for tok in val.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out if out else None


def format_save_dir(save_dir):
    if not isinstance(save_dir, str):
        return save_dir
    try:
        return save_dir.format(now=datetime.now())
    except Exception:
        return save_dir


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def set_axes_box_size_mm(w_mm, h_mm, margins_mm=(15, 5, 5, 5), fig=None, ax=None):
    """
    Set the axes box (data area) to an exact size in mm.

    If fig/ax are not given, use current figure and axes.
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    mm_per_in = 25.4
    w_in = w_mm / mm_per_in
    h_in = h_mm / mm_per_in
    ml, mr, mt, mb = (m / mm_per_in for m in margins_mm)

    fig_w = w_in + ml + mr
    fig_h = h_in + mt + mb
    fig.set_size_inches(fig_w, fig_h)

    left = ml / fig_w
    bottom = mb / fig_h
    width = w_in / fig_w
    height = h_in / fig_h
    ax.set_position([left, bottom, width, height])

    return fig, ax


def _read_ds(group, name):
    if name not in group:
        return None
    ds = group[name]
    enc = ds.attrs.get("encoding", None)
    if enc == b"json-utf8" or enc == "json-utf8":
        payload = bytes(ds[:].tolist())
        return np.array(json.loads(payload.decode("utf-8")))
    return ds[:]


def load_final_dirs_by_angle(h5_path):
    angles_to_finaldirs = {}
    with h5py.File(h5_path, "r") as f:
        runs = f["runs"]
        for run_name in runs:
            g = runs[run_name]
            if "angle" not in g.attrs:
                continue
            angle = float(g.attrs["angle"])
            avg_dir = _read_ds(g, "average_direction")
            if avg_dir is None or len(avg_dir) == 0:
                continue
            final_direction = float(avg_dir[-1])
            angles_to_finaldirs.setdefault(angle, []).append(final_direction)
    return angles_to_finaldirs


def plot_static_decision_scatter(
    h5_path,
    out_path,
    jitter_scale=0.03,
    seed=0,
    dpi=300,
    transparent=True,
    bbox_inches="tight",
    show=False,
    close_fig=True,
    axes_box_mm=(50.755, 22.861),
    axes_margins_mm=(18, 8, 6, 12),
):
    angles_to_finaldirs = load_final_dirs_by_angle(h5_path)
    print(f"Loaded {len(angles_to_finaldirs)} unique angles from {h5_path}")
    if not angles_to_finaldirs:
        raise RuntimeError("No data found. Check that 'average_direction' exists in each run.")

    sorted_angles_rad = sorted(angles_to_finaldirs.keys())
    fig, ax = plt.subplots(figsize=(4, 2))
    set_axes_box_size_mm(
        w_mm=axes_box_mm[0],
        h_mm=axes_box_mm[1],
        margins_mm=axes_margins_mm,
        fig=fig,
        ax=ax,
    )
    rng = np.random.default_rng(seed)

    for angle_rad in sorted_angles_rad:
        final_dirs_rad = np.asarray(angles_to_finaldirs[angle_rad], dtype=float)
        x_jitter = rng.normal(loc=angle_rad, scale=jitter_scale, size=len(final_dirs_rad))
        ax.scatter(x_jitter, final_dirs_rad, alpha=0.3, color=(19 / 255, 62 / 255, 118 / 255))

    xmin_rad = min(sorted_angles_rad)
    xmax_rad = max(sorted_angles_rad)
    x_line_rad = np.linspace(xmin_rad, xmax_rad, 200)
    ax.plot(x_line_rad, 0.5 * x_line_rad, "-", color="grey", alpha=0.3)
    ax.plot(x_line_rad, -0.5 * x_line_rad, "-", color="grey", alpha=0.3, label="Direction of goals")

    ax.set_xlabel(r"angular difference")
    ax.set_ylabel(r"angles $\theta$")
    ax.grid(False)

    ax.set_xticks([0, np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi$"])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, transparent=transparent, bbox_inches=bbox_inches)
    if show:
        plt.show()
    if close_fig:
        plt.close(fig)
    return str(out_path)

def results_from_sim(sim):
    """Collect results from a finished Simulation into a dict of numpy arrays."""

    def to_np(x, dtype=None):
        try:
            return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)
        except Exception:
            return np.array(x, dtype=object)

    return {
        "agent_positions": to_np(sim.agent_positions),            # may be tiny
        "spin_states":     to_np(sim.spin_states),                # may be huge → optional
        "goals":           to_np(sim.goals),                      # likely object → JSON branch
        "goal_reached":    np.array(sim.reached_goal_index),
        "num_groups":      np.array(sim.spin_system.num_groups),
        "num_spins_per_group": np.array(sim.spin_system.num_spins_per_group),
        "width":           to_np(sim.width, dtype=float),         # force numeric series
        "average_direction": to_np(sim.direction, dtype=float),   # force numeric series
        "heading":         np.array(sim.agent.orientation),
    }



# ==========================
# Worker
# ==========================

def run_static_decision_experiment(params, num_steps, run_id,
                                   seed=None, angle=None, store_spin_states=True):
    """
    A single static decision run that RETURNS results (no file writing here).
    """
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            import torch
            torch.manual_seed(seed)
    except Exception:
        pass

    sim = Simulation(**params)
    sim.run_static_simulation(num_steps=num_steps)

    results = results_from_sim(sim)
    if not store_spin_states and "spin_states" in results:
        del results["spin_states"]

    return {
        "run_id": run_id,
        "angle": float(angle) if angle is not None else None,
        "results": results,
        "per_run_params": {"goals": params.get("goals", [])},
    }


# ==========================
# Orchestrator
# ==========================

def run_static_decision_experiments(base_params, num_steps, num_runs, save_dir,
                                    store_spin_states=False, max_workers=None, kappa_values=None,
                                    strength_values=None, angles=None, base_seed=12345):
    """
    Build tasks → run in parallel → parent writes all results into a single HDF5 file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save a copy of this script (for reproducibility)
    script_path = os.path.abspath(sys.argv[0])
    try:
        shutil.copy(script_path, os.path.join(save_dir, os.path.basename(script_path)))
    except Exception:
        pass

    # Prepare HDF5 writer
    h5_path = os.path.join(save_dir, "experiment.h5")
    writer = H5ExperimentWriter(
        h5_path,
        base_params=base_params,
        runtime_meta={
            "version": code_fingerprint(),
            "one_word_description": f"sweep_angles_symmetric_kappa_{kappa_values[0]}_{kappa_values[1]}_strength_{strength_values[0]}_{strength_values[1]}_prerun",
            "created_by": "static_decision_sweep",
        },
        compression="gzip",
        compression_lvl=4,
    )

    # Build tasks
    tasks = []
    run_counter = 0
    for angle in angles:
        new_goals = [(-angle / 2, 10, strength_values[0], kappa_values[0]),
                     ( angle / 2, 10, strength_values[1], kappa_values[1])]

        for r in range(num_runs):
            seed = base_seed + r + int(angle * 1000)
            params = base_params.copy()
            params["goals"] = new_goals
            tasks.append((params, num_steps, run_counter, seed, angle))
            run_counter += 1

    # Execute in parallel; parent writes sequentially
    futures = []
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor, \
        tqdm(total=total, desc="Static decision runs", unit="run", dynamic_ncols=True) as pbar:

        for (params, steps, run_id, seed, angle) in tasks:
            futures.append(
                executor.submit(
                    run_static_decision_experiment,
                    params, steps, run_id,
                    seed=seed, angle=angle,
                    store_spin_states=store_spin_states,
                )
            )

        for fut in as_completed(futures):
            out = fut.result()  # re-raises any worker exception
            writer.write_group(
                name=f"run_{out['run_id']:05d}_angle_{out['angle']:.4f}",
                attrs={
                    "run_id": out["run_id"],
                    "angle": out["angle"],
                    "per_run_params": out["per_run_params"],
                },
                datasets=out["results"],
                skip_keys={"spin_states"} if not store_spin_states else None,
            )
            # update progress
            pbar.update(1)
            # optional: show last angle/run in the bar
            pbar.set_postfix(run=out["run_id"], angle=f"{out['angle']:.4f}")


    print(f"All runs completed. Saved to: {h5_path}")
    return h5_path


def resolve_angles(cfg):
    if cfg.get("angles_rad_min") is not None and cfg.get("angles_rad_max") is not None and cfg.get("angles_rad_num"):
        return np.linspace(cfg["angles_rad_min"], cfg["angles_rad_max"], int(cfg["angles_rad_num"]))
    if cfg.get("angles_deg_min") is not None and cfg.get("angles_deg_max") is not None and cfg.get("angles_deg_num"):
        return np.deg2rad(np.linspace(cfg["angles_deg_min"], cfg["angles_deg_max"], int(cfg["angles_deg_num"])))
    angles_deg = cfg.get("angles_deg")
    if angles_deg is not None and len(angles_deg) > 0:
        return np.deg2rad(np.array(angles_deg, dtype=float))
    angles_rad = cfg.get("angles_rad")
    if angles_rad is not None and len(angles_rad) > 0:
        return np.array(angles_rad, dtype=float)
    return np.linspace(0, math.pi, 185)


# ==========================
# Main
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Run static decision angle sweeps (HDF5 output) and plot Figure S4C-style scatter."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--num-runs", type=int, default=None, help="Runs per angle.")
    parser.add_argument("--num-steps", type=int, default=None, help="Steps per run.")
    parser.add_argument("--angles-rad", type=str, default=None, help="Comma-separated angles (radians).")
    parser.add_argument("--angles-deg", type=str, default=None, help="Comma-separated angles (degrees).")
    parser.add_argument("--kappa-values", type=str, default=None, help="Comma-separated kappas (len 2).")
    parser.add_argument("--strength-values", type=str, default=None, help="Comma-separated strengths (len 2).")
    parser.add_argument("--save-dir", type=str, default=None, help="Where to write results.")
    parser.add_argument("--store-spin-states", action="store_true", help="Store spin_states (large).")
    parser.add_argument("--max-workers", type=int, default=None, help="ProcessPoolExecutor workers.")
    parser.add_argument("--no-plot", action="store_true", help="Disable automatic plotting.")
    parser.add_argument("--plot-only", type=str, default=None, help="Path to existing HDF5; only plot and exit.")
    parser.add_argument("--plot-filename", type=str, default=None, help="Output filename for the plot.")
    args = parser.parse_args()

    cfg = {
        "base_params": default_base_params(),
        "num_runs": 1,
        "num_steps": 1000,
        "angles_rad": list(np.linspace(0, math.pi, 185)),
        "angles_rad_min": None,
        "angles_rad_max": None,
        "angles_rad_num": None,
        "angles_deg": None,
        "angles_deg_min": None,
        "angles_deg_max": None,
        "angles_deg_num": None,
        "kappa_values": [10.0, 10.0],
        "strength_values": [1.0, 1.0],
        "save_dir": f"/Users/apaula/src/SimulationResults/{{now:%Y%m%d_%H%M%S}}_sweep_angles_symmetric_kappa_10_10_strength_1_1_prerun",
        "store_spin_states": False,
        "max_workers": None,
        "base_seed": 12345,
        "plot_enabled": True,
        "plot_filename": "FigureS4C.svg",
        "plot_dpi": 300,
        "plot_transparent": True,
        "plot_bbox_inches": "tight",
        "plot_jitter_scale": 0.03,
        "plot_seed": 0,
        "plot_show": False,
        "plot_axes_box_mm": [50.755, 22.861],
        "plot_axes_margins_mm": [18, 8, 6, 12],
        "plot_save_dir": None,
    }

    if args.config:
        file_cfg = load_config(args.config)
        cfg.update({k: v for k, v in file_cfg.items() if v is not None})
        if "base_params" not in cfg or not cfg["base_params"]:
            cfg["base_params"] = default_base_params()

    if args.num_runs is not None:
        cfg["num_runs"] = args.num_runs
    if args.num_steps is not None:
        cfg["num_steps"] = args.num_steps
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    if args.store_spin_states:
        cfg["store_spin_states"] = True
    if args.max_workers is not None:
        cfg["max_workers"] = args.max_workers
    if args.no_plot:
        cfg["plot_enabled"] = False
    if args.plot_filename is not None:
        cfg["plot_filename"] = args.plot_filename
    if args.plot_only is not None:
        cfg["plot_only"] = args.plot_only
    ang_rad_override = parse_csv_floats(args.angles_rad)
    if ang_rad_override is not None:
        cfg["angles_rad"] = ang_rad_override
    ang_deg_override = parse_csv_floats(args.angles_deg)
    if ang_deg_override is not None:
        cfg["angles_deg"] = ang_deg_override
    kappa_override = parse_csv_floats(args.kappa_values)
    if kappa_override is not None:
        cfg["kappa_values"] = kappa_override
    strength_override = parse_csv_floats(args.strength_values)
    if strength_override is not None:
        cfg["strength_values"] = strength_override

    cfg["save_dir"] = format_save_dir(cfg["save_dir"])
    angles = resolve_angles(cfg)
    cfg["angles_rad"] = [float(v) for v in angles]

    plot_only = cfg.get("plot_only")
    if plot_only:
        h5_path = plot_only
        plot_save_dir = cfg.get("plot_save_dir") or os.path.dirname(h5_path)
        plot_filename = cfg.get("plot_filename") or "FigureS4C.svg"
        plot_path = plot_filename if os.path.isabs(plot_filename) else os.path.join(plot_save_dir, plot_filename)
        plot_static_decision_scatter(
            h5_path=h5_path,
            out_path=plot_path,
            jitter_scale=cfg.get("plot_jitter_scale", 0.03),
            seed=cfg.get("plot_seed", 0),
            dpi=cfg.get("plot_dpi", 300),
            transparent=cfg.get("plot_transparent", True),
            bbox_inches=cfg.get("plot_bbox_inches", "tight"),
            show=cfg.get("plot_show", False),
            axes_box_mm=tuple(cfg.get("plot_axes_box_mm", [50.755, 22.861])),
            axes_margins_mm=tuple(cfg.get("plot_axes_margins_mm", [18, 8, 6, 12])),
        )
        return

    h5_path = run_static_decision_experiments(
        cfg["base_params"],
        cfg["num_steps"],
        cfg["num_runs"],
        cfg["save_dir"],
        store_spin_states=cfg["store_spin_states"],
        max_workers=cfg["max_workers"],
        kappa_values=cfg["kappa_values"],
        strength_values=cfg["strength_values"],
        angles=angles,
        base_seed=cfg.get("base_seed", 12345),
    )

    if cfg.get("plot_enabled", True):
        plot_save_dir = cfg.get("plot_save_dir") or cfg["save_dir"]
        plot_filename = cfg.get("plot_filename") or "FigureS4C.svg"
        plot_path = plot_filename if os.path.isabs(plot_filename) else os.path.join(plot_save_dir, plot_filename)
        plot_static_decision_scatter(
            h5_path=h5_path,
            out_path=plot_path,
            jitter_scale=cfg.get("plot_jitter_scale", 0.03),
            seed=cfg.get("plot_seed", 0),
            dpi=cfg.get("plot_dpi", 300),
            transparent=cfg.get("plot_transparent", True),
            bbox_inches=cfg.get("plot_bbox_inches", "tight"),
            show=cfg.get("plot_show", False),
            axes_box_mm=tuple(cfg.get("plot_axes_box_mm", [50.755, 22.861])),
            axes_margins_mm=tuple(cfg.get("plot_axes_margins_mm", [18, 8, 6, 12])),
        )


if __name__ == "__main__":
    main()
