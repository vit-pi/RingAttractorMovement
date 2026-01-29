import os
import json
import math
import shutil
import random
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.simulation import Simulation
from core.experiment_utils import code_fingerprint, H5ExperimentWriter


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def _detect_sweep_var(runs_grp):
    has_strength = any("strength" in runs_grp[name].attrs for name in runs_grp)
    has_kappa = any("kappa" in runs_grp[name].attrs for name in runs_grp)
    if has_strength:
        return "strength"
    if has_kappa:
        return "kappa"
    return "kappa"


def load_final_dirs_by_angle_var(h5_file, var_name=None):
    data_dict = {}
    with h5py.File(h5_file, "r") as f:
        runs_grp = f["runs"]
        sweep_var = var_name or _detect_sweep_var(runs_grp)

        for run_name in runs_grp:
            g = runs_grp[run_name]
            if "angle" not in g.attrs or sweep_var not in g.attrs:
                continue
            angle = float(g.attrs["angle"])
            var_val = float(g.attrs[sweep_var])

            if "average_direction" not in g:
                continue
            avg_dir = np.array(g["average_direction"])
            final_dir = float(avg_dir[-1]) if avg_dir.size > 0 else np.nan
            data_dict.setdefault((angle, var_val), []).append(final_dir)

    angles_sorted = sorted({a for (a, _) in data_dict.keys()})
    values_sorted_per_angle = {
        a: sorted({v for (aa, v) in data_dict.keys() if np.isclose(aa, a)})
        for a in angles_sorted
    }
    return sweep_var, data_dict, angles_sorted, values_sorted_per_angle


def set_axes_box_size_mm(w_mm, h_mm, margins_mm=(15, 5, 5, 5), fig=None, ax=None):
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


def format_save_dir(save_dir):
    if not isinstance(save_dir, str):
        return save_dir
    try:
        return save_dir.format(now=datetime.now())
    except Exception:
        return save_dir


def _to_np(value, dtype=None):
    try:
        return np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    except Exception:
        return np.array(value, dtype=object)


def results_from_sim(sim):
    return {
        "agent_positions": _to_np(sim.agent_positions),
        "spin_states": _to_np(sim.spin_states),
        "goals": _to_np(sim.goals),
        "goal_reached": np.array(sim.reached_goal_index),
        "num_groups": np.array(sim.spin_system.num_groups),
        "num_spins_per_group": np.array(sim.spin_system.num_spins_per_group),
        "width": _to_np(sim.width, dtype=float),
        "average_direction": _to_np(sim.direction, dtype=float),
        "heading": np.array(sim.agent.orientation),
    }


def run_static_decision_experiment(params, num_steps, run_id,
                                   seed=None, angle=None, sweep_val=None, store_spin_states=True):
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
        "sweep_val": float(sweep_val) if sweep_val is not None else None,
        "results": results,
        "per_run_params": {"goals": params.get("goals", [])},
    }


def _resolve_angle(cfg):
    if cfg.get("angle_rad") is not None:
        return float(cfg["angle_rad"])
    if cfg.get("angle_deg") is not None:
        return float(cfg["angle_deg"]) * math.pi / 180.0
    return None


def _resolve_sweep_var(cfg):
    if cfg.get("sweep_var") in {"kappa", "strength"}:
        return cfg["sweep_var"]
    if cfg.get("kappa_values") is not None:
        return "kappa"
    if cfg.get("strength_values") is not None:
        return "strength"
    return None


def run_fixed_angle_sweep(cfg, config_path=None):
    sweep_var = _resolve_sweep_var(cfg)
    if sweep_var is None:
        raise SystemExit("Missing sweep_var or kappa_values/strength_values in config.")

    angle = _resolve_angle(cfg)
    if angle is None:
        raise SystemExit("Missing angle_rad or angle_deg in config.")

    values = cfg.get("kappa_values") if sweep_var == "kappa" else cfg.get("strength_values")
    if not values:
        raise SystemExit(f"Missing {sweep_var}_values in config.")
    values = [float(v) for v in values]

    base_params = cfg.get("base_params") or {}
    num_steps = int(cfg.get("num_steps", 1000))
    num_runs = int(cfg.get("num_runs", 1))
    save_dir = cfg.get("save_dir")
    if not save_dir:
        raise SystemExit("Missing save_dir in config for running simulation.")
    save_dir = format_save_dir(save_dir)
    cfg["save_dir"] = save_dir

    store_spin_states = bool(cfg.get("store_spin_states", False))
    max_workers = cfg.get("max_workers")
    base_seed = int(cfg.get("base_seed", 12345))

    goal1_cfg = cfg.get("fixed_goal1", {}) or {}
    goal2_cfg = cfg.get("fixed_goal2", {}) or {}
    g1_radius = float(goal1_cfg.get("radius", 10.0))
    g1_strength = float(goal1_cfg.get("strength", 1.0))
    g1_kappa = float(goal1_cfg.get("kappa", 10.0))
    g2_radius = float(goal2_cfg.get("radius", 10.0))
    g2_strength = float(goal2_cfg.get("strength", 1.0))
    g2_kappa = float(goal2_cfg.get("kappa", 10.0))

    os.makedirs(save_dir, exist_ok=True)

    h5_path = os.path.join(save_dir, "experiment.h5")
    writer = H5ExperimentWriter(
        h5_path,
        base_params=base_params,
        runtime_meta={
            "version": code_fingerprint(),
            "description": f"fixed_angle_sweep_{sweep_var}_goal2",
            "angle": float(angle),
        },
        compression="gzip",
        compression_lvl=4,
    )

    tasks = []
    run_counter = 0
    for sweep_val in values:
        if sweep_var == "kappa":
            goal2_kappa = float(sweep_val)
            goal2_strength = g2_strength
        else:
            goal2_strength = float(sweep_val)
            goal2_kappa = g2_kappa

        new_goals = [
            (-angle / 2, g1_radius, g1_strength, g1_kappa),
            ( angle / 2, g2_radius, goal2_strength, goal2_kappa),
        ]
        for r in range(num_runs):
            seed = base_seed + r + int((float(sweep_val) + 0.5) * 1000)
            params = base_params.copy()
            params["goals"] = new_goals
            tasks.append((params, num_steps, run_counter, seed, angle, float(sweep_val)))
            run_counter += 1

    futures = []
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor, \
        tqdm(total=total, desc="Static decision runs", unit="run", dynamic_ncols=True) as pbar:

        for (params, steps, run_id, seed, ang, sweep_val) in tasks:
            futures.append(
                executor.submit(
                    run_static_decision_experiment,
                    params, steps, run_id,
                    seed=seed, angle=ang, sweep_val=sweep_val,
                    store_spin_states=store_spin_states,
                )
            )

        for fut in as_completed(futures):
            out = fut.result()
            writer.write_group(
                name=f"run_{out['run_id']:05d}_ang_{out['angle']:.4f}_{sweep_var}_{out['sweep_val']:.4f}",
                attrs={
                    "run_id": out["run_id"],
                    "angle": float(out["angle"]),
                    sweep_var: float(out["sweep_val"]),
                    "per_run_params": out["per_run_params"],
                },
                datasets=out["results"],
                skip_keys={"spin_states"} if not store_spin_states else None,
            )
            pbar.update(1)
            pbar.set_postfix(run=out["run_id"], sweep=f"{out['sweep_val']:.4f}")

    print(f"All runs completed. Saved to: {h5_path}")
    cfg["h5_path"] = h5_path
    return h5_path


def plot_bars_with_points_proportional(
    sweep_var,
    data_dict,
    angles_sorted,
    values_sorted_per_angle,
    angle_selection=None,
    jitter_scale=0.03,
    save_path=None,
    dpi=300,
    transparent=True,
    bbox_inches="tight",
    figsize=(4, 2),
    axes_box_mm=(50.755, 22.861),
    axes_margins_mm=(18, 8, 6, 12),
    return_fig=False,
    close_fig=True,
):
    label = "Inverse Width ($\\kappa$)" if sweep_var == "kappa" else "Strength ($s_0$)"
    angles_to_plot = angles_sorted if angle_selection is None else [angle_selection]
    rng = np.random.default_rng(0)

    figs_axes = []

    for angle_val in angles_to_plot:
        var_to_dirs = {
            v: data_dict[(a, v)]
            for (a, v) in data_dict.keys()
            if np.isclose(a, angle_val)
        }
        sorted_vals = values_sorted_per_angle.get(angle_val, [])
        if not sorted_vals:
            continue

        neg_counts, pos_counts, totals = [], [], []
        for v in sorted_vals:
            ys = np.asarray(var_to_dirs.get(v, []), dtype=float)
            ys = ys[~np.isnan(ys)]
            nneg = int(np.sum(ys < 0.0))
            npos = int(np.sum(ys >= 0.0))
            total = nneg + npos
            neg_counts.append(nneg)
            pos_counts.append(npos)
            totals.append(total if total > 0 else 1)

        x = np.arange(len(sorted_vals))
        fig, ax = plt.subplots(figsize=figsize)
        set_axes_box_size_mm(
            w_mm=axes_box_mm[0],
            h_mm=axes_box_mm[1],
            margins_mm=axes_margins_mm,
            fig=fig,
            ax=ax,
        )

        y_min, y_max = -np.pi / 2, np.pi / 2
        ax.set_ylim(y_min, y_max)

        total_span = (y_max - y_min)
        neg_fracs = [nc / t for nc, t in zip(neg_counts, totals)]
        pos_fracs = [pc / t for pc, t in zip(pos_counts, totals)]
        split_ys = [y_min + total_span * nf for nf in neg_fracs]

        ax.bar(
            x,
            [total_span * nf for nf in neg_fracs],
            bottom=y_min,
            alpha=0.5,
            color="grey",
        )
        ax.bar(
            x,
            [total_span * pf for pf in pos_fracs],
            bottom=split_ys,
            alpha=0.5,
            color=(211 / 255, 170 / 255, 2 / 255),
        )

        for xi, v in zip(x, sorted_vals):
            ys = np.asarray(var_to_dirs.get(v, []), dtype=float)
            ys = ys[~np.isnan(ys)]
            if ys.size == 0:
                continue
            xs_jitter = rng.normal(loc=xi, scale=jitter_scale, size=ys.size)
            ax.scatter(
                xs_jitter,
                ys,
                alpha=0.3,
                edgecolors="white",
                linewidth=0.5,
                zorder=3,
                color=(19 / 255, 62 / 255, 118 / 255),
            )

        ax.set_ylabel(r"angles $\theta$")
        ax.set_yticks([-np.pi * 1 / 3, 0, np.pi * 1 / 3])
        ax.set_yticklabels([r"$-\dfrac{\pi}{3}$", r"$0$", r"$\dfrac{\pi}{3}$"])

        ax.set_xticks(x[::2])
        ax.set_xticklabels(
            [f"{int(v)}" if float(v).is_integer() else f"{v:.1f}" for v in sorted_vals[::2]],
            rotation=45,
        )
        ax.set_xlabel(label)

        ax.axhline(y=angle_val / 2, linestyle="-", color=(211 / 255, 170 / 255, 2 / 255), alpha=0.3)
        ax.axhline(y=-angle_val / 2, linestyle="-", color="grey", alpha=0.3)

        if save_path is not None:
            save_path = Path(save_path)
            if len(angles_to_plot) > 1:
                stem = save_path.stem
                save_path = save_path.with_stem(f"{stem}_angle_{float(angle_val):.3f}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_path,
                dpi=dpi,
                transparent=transparent,
                bbox_inches=bbox_inches,
            )

        figs_axes.append((fig, ax))
        if close_fig:
            plt.close(fig)

    if return_fig:
        return figs_axes if len(figs_axes) != 1 else figs_axes[0]


def _resolve_angle_selection(cfg):
    if cfg.get("angle_selection_rad") is not None:
        return float(cfg["angle_selection_rad"])
    if cfg.get("angle_selection_deg") is not None:
        return float(cfg["angle_selection_deg"]) * np.pi / 180.0
    return None


def _resolve_save_path(cfg, h5_path):
    filename = cfg.get("plot_filename") or cfg.get("save_path") or "figureS4G.svg"
    if os.path.isabs(filename):
        return filename
    plot_dir = cfg.get("plot_save_dir") or cfg.get("save_dir") or os.path.dirname(h5_path) or "."
    return os.path.join(plot_dir, filename)


def main():
    parser = argparse.ArgumentParser(
        description="Plot S4G/S4K proportional bar+point figure from an HDF5 sweep."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    parser.add_argument("--h5-path", type=str, default=None, help="Override HDF5 path.")
    parser.add_argument("--plot-filename", type=str, default=None, help="Override output filename.")
    parser.add_argument("--sweep-var", type=str, default=None, choices=["kappa", "strength"],
                        help="Override sweep variable (kappa/strength).")
    parser.add_argument("--angle-selection-rad", type=float, default=None, help="Plot only this angle (rad).")
    parser.add_argument("--angle-selection-deg", type=float, default=None, help="Plot only this angle (deg).")
    parser.add_argument("--run", action="store_true", help="Run sweep before plotting.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.h5_path is not None:
        cfg["h5_path"] = args.h5_path
    if args.plot_filename is not None:
        cfg["plot_filename"] = args.plot_filename
    if args.sweep_var is not None:
        cfg["sweep_var"] = args.sweep_var
    if args.angle_selection_rad is not None:
        cfg["angle_selection_rad"] = args.angle_selection_rad
    if args.angle_selection_deg is not None:
        cfg["angle_selection_deg"] = args.angle_selection_deg
    if args.run:
        cfg["run_simulation"] = True

    h5_path = cfg.get("h5_path")
    if cfg.get("run_simulation") or not h5_path:
        h5_path = run_fixed_angle_sweep(cfg, config_path=args.config)
    if not h5_path:
        raise SystemExit("Missing h5_path in config and run_simulation is false.")

    sweep_var = cfg.get("sweep_var")
    sweep_var, data_dict, angles_sorted, values_sorted_per_angle = load_final_dirs_by_angle_var(
        h5_path, var_name=sweep_var
    )

    total_runs = sum(len(v) for v in data_dict.values())
    print(f"[{sweep_var}] Loaded {total_runs} runs across {len(angles_sorted)} angle(s).")

    angle_selection = _resolve_angle_selection(cfg)
    save_path = _resolve_save_path(cfg, h5_path)

    plot_bars_with_points_proportional(
        sweep_var,
        data_dict,
        angles_sorted,
        values_sorted_per_angle,
        angle_selection=angle_selection,
        jitter_scale=cfg.get("plot_jitter_scale", 0.03),
        save_path=save_path,
        dpi=cfg.get("plot_dpi", 300),
        transparent=cfg.get("plot_transparent", True),
        bbox_inches=cfg.get("plot_bbox_inches", "tight"),
        axes_box_mm=tuple(cfg.get("plot_axes_box_mm", [50.755, 22.861])),
        axes_margins_mm=tuple(cfg.get("plot_axes_margins_mm", [18, 8, 6, 12])),
        return_fig=False,
        close_fig=True,
    )

    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    main()
