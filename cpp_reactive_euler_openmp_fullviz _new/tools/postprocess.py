from __future__ import annotations

import glob
import json
import math
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    imageio = None
    IMAGEIO_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    Image = None
    PIL_AVAILABLE = False

NV = 5
MAGIC = 20260527
HEADER_FMT = "iii7d"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

FIGSIZE_PAIR = (14.0, 7.8)
FIGSIZE_PAIR_TALL = (14.0, 8.4)
FIGSIZE_YAVG_FULL = (10.0, 12.0)

CMAP_SCALAR = "viridis"
CMAP_DIVERGING = "RdBu_r"
CMAP_SCHLIEREN = "magma"
CMAP_T = "inferno"


@dataclass
class Params:
    gamma: float = 1.4
    Q: float = 20.0
    k: float = 1.0
    Ea: float = 20.0
    R: float = 1.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return str(path)


def safe_clean_dir(path):
    path = str(path)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


def cons_to_prim_2d(U, gamma=1.4, Q=20.0):
    U = np.asarray(U)
    rho = np.maximum(U[..., 0], 1.0e-10)
    mx = U[..., 1]
    my = U[..., 2]
    E = U[..., 3]
    rlam = U[..., 4]
    u = mx / rho
    v = my / rho
    lam = np.clip(rlam / rho, 0.0, 1.0)
    kinetic = 0.5 * (mx * mx + my * my) / rho
    p = (gamma - 1.0) * (E - kinetic + Q * rlam)
    p = np.maximum(p, 1.0e-10)
    return rho, u, v, p, lam


def primitive_fields_from_U(Uinner, par: Params):
    rho, u, v, p, lam = cons_to_prim_2d(Uinner, par.gamma, par.Q)
    T = p / np.maximum(rho * par.R, 1.0e-14)
    return rho, u, v, p, T, lam


def y_average_from_U(Uinner, par: Params):
    rho, u, v, p, T, lam = primitive_fields_from_U(Uinner, par)
    return {
        "rho": np.mean(rho, axis=1),
        "u": np.mean(u, axis=1),
        "v": np.mean(v, axis=1),
        "p": np.mean(p, axis=1),
        "T": np.mean(T, axis=1),
        "lam": np.mean(lam, axis=1),
    }


def read_cpp_snapshot_bin(path):
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise ValueError(f"Bad snapshot")
        magic, Nx, Ny, t, xL, xR, yL, yR, dx, dy = struct.unpack(HEADER_FMT, header)
        if magic != MAGIC:
            raise ValueError(f"Bad snapshot")
        n_u = Nx * Ny * NV
        U = np.frombuffer(f.read(n_u * 8), dtype=np.float64).copy()
        if U.size != n_u:
            raise ValueError(f"Truncated U")
        U = U.reshape((Nx, Ny, NV))
        rest = np.frombuffer(f.read(), dtype=np.float64).copy()

    p_max = None
    T_max = None
    if rest.size >= 2 * Nx * Ny:
        p_max = rest[: Nx * Ny].reshape((Nx, Ny))
        T_max = rest[Nx * Ny : 2 * Nx * Ny].reshape((Nx, Ny))

    x = xL + (np.arange(Nx, dtype=float) + 0.5) * dx
    y = yL + (np.arange(Ny, dtype=float) + 0.5) * dy
    return {
        "t": float(t),
        "x": x,
        "y": y,
        "U": U,
        "p_max": p_max,
        "T_max": T_max,
        "Nx": Nx,
        "Ny": Ny,
        "domain": (xL, xR, yL, yR),
        "dx": dx,
        "dy": dy,
        "has_extrema_payload": p_max is not None and T_max is not None,
    }


def list_cpp_binary_snapshots(snapshot_dir):
    return sorted(Path(snapshot_dir).glob("snapshot_*.bin"))


def convert_cpp_binary_snapshots_to_npz(snapshot_bin_dir, snapshot_npz_dir, par: Params, dtype=np.float32, overwrite=True):
    snapshot_bin_dir = Path(snapshot_bin_dir)
    snapshot_npz_dir = Path(snapshot_npz_dir)
    if overwrite and snapshot_npz_dir.exists():
        shutil.rmtree(snapshot_npz_dir)
    snapshot_npz_dir.mkdir(parents=True, exist_ok=True)

    files = list_cpp_binary_snapshots(snapshot_bin_dir)
    if not files:
        raise RuntimeError(f"No C++ snapshot_*.bin")

    manifest = []
    p_cum = None
    T_cum = None
    used_cpp_extrema = False

    for sid, f in enumerate(files):
        snap = read_cpp_snapshot_bin(f)
        U = snap["U"]
        rho, u, v, p, T, lam = primitive_fields_from_U(U, par)

        if snap["p_max"] is not None and snap["T_max"] is not None:
            p_max = np.asarray(snap["p_max"], dtype=float)
            T_max = np.asarray(snap["T_max"], dtype=float)
            used_cpp_extrema = True
        else:
            if p_cum is None:
                p_cum = p.copy()
                T_cum = T.copy()
            else:
                np.maximum(p_cum, p, out=p_cum)
                np.maximum(T_cum, T, out=T_cum)
            p_max = p_cum
            T_max = T_cum

        out = snapshot_npz_dir / f"snapshot_{sid:05d}.npz"
        np.savez(
            out,
            t=np.array(float(snap["t"]), dtype=np.float64),
            x=snap["x"].astype(np.float32, copy=False),
            y=snap["y"].astype(np.float32, copy=False),
            U=U.astype(dtype, copy=False),
            p_max=p_max.astype(dtype, copy=False),
            T_max=T_max.astype(dtype, copy=False),
        )
        manifest.append({"id": int(sid), "t": float(snap["t"]), "file": str(out)})

    with (snapshot_npz_dir / "manifest.json").open("w") as f:
        json.dump({"snapshots": manifest, "used_cpp_extrema": bool(used_cpp_extrema)}, f, indent=2)

    return manifest


def list_disk_snapshots(snapshot_dir):
    return sorted(glob.glob(os.path.join(str(snapshot_dir), "snapshot_*.npz")))


def load_disk_snapshot(snapshot_file):
    data = np.load(snapshot_file)
    snap = {
        "t": float(data["t"]),
        "x": np.asarray(data["x"], dtype=float),
        "y": np.asarray(data["y"], dtype=float),
        "U": np.asarray(data["U"], dtype=float),
    }
    snap["p_max"] = np.asarray(data["p_max"], dtype=float) if "p_max" in data.files else None
    snap["T_max"] = np.asarray(data["T_max"], dtype=float) if "T_max" in data.files else None
    return snap


def schlieren_from_fields(x, y, rho):
    edge_order = 2 if len(x) >= 3 and len(y) >= 3 else 1
    drho_dx, drho_dy = np.gradient(rho, x, y, edge_order=edge_order)
    grad = np.sqrt(drho_dx**2 + drho_dy**2)
    return np.log1p(grad)


def vorticity_from_fields(x, y, u, v):
    edge_order = 2 if len(x) >= 3 and len(y) >= 3 else 1
    du_dx, du_dy = np.gradient(u, x, y, edge_order=edge_order)
    dv_dx, dv_dy = np.gradient(v, x, y, edge_order=edge_order)
    return dv_dx - du_dy


def contrast_y(field_xy):
    mean_y = np.nanmean(field_xy, axis=1, keepdims=True)
    return field_xy / (mean_y + 1.0e-14) - 1.0


def robust_limits(values, qlo=1.0, qhi=99.0, symmetric=False, pad=0.05):
    arr = np.asarray(values)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        vmax = np.nanpercentile(np.abs(arr), qhi)
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = np.nanmax(np.abs(arr)) + 1.0e-14
        return (-vmax, vmax)
    lo = np.nanpercentile(arr, qlo)
    hi = np.nanpercentile(arr, qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = np.nanmin(arr)
        hi = np.nanmax(arr)
        if lo == hi:
            lo -= 1.0
            hi += 1.0
    width = hi - lo
    return (lo - pad * width, hi + pad * width)


def save_fixed_size_figure(fig, out_file, dpi=120):
    out_file = str(out_file)
    ensure_dir(os.path.dirname(out_file) or ".")
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)
    return out_file


def get_fixed_xlim(x, fixed_xlim=None):
    if fixed_xlim is None:
        return (float(x[0]), float(x[-1]))
    return (float(fixed_xlim[0]), float(fixed_xlim[1]))


def x_mask_from_xlim(x, xlim):
    mask = (x >= xlim[0]) & (x <= xlim[1])
    if not np.any(mask):
        return np.ones_like(x, dtype=bool)
    return mask


class RecentWindowMaxCache:
    def __init__(self, files, par, window=2.5):
        self.files = list(files)
        self.par = par
        self.window = float(window)
        self.times = np.array([load_disk_snapshot(f)["t"] for f in self.files], dtype=float)
        self._p_cache = {}

    def _pressure_for_index(self, idx):
        if idx not in self._p_cache:
            snap = load_disk_snapshot(self.files[idx])
            rho, u, v, p, T, lam = primitive_fields_from_U(snap["U"], self.par)
            self._p_cache[idx] = p
            if len(self._p_cache) > 20:
                for old_key in sorted(self._p_cache.keys())[:-20]:
                    self._p_cache.pop(old_key, None)
        return self._p_cache[idx]

    def p_recent_for_index(self, idx):
        t = self.times[idx]
        j0 = int(np.searchsorted(self.times, t - self.window - 1.0e-12, side="left"))
        j1 = idx
        p_recent = None
        for j in range(j0, j1 + 1):
            p_j = self._pressure_for_index(j)
            if p_recent is None:
                p_recent = p_j.copy()
            else:
                np.maximum(p_recent, p_j, out=p_recent)
        return p_recent


def compute_animation_limits(snapshot_dir, par, fixed_xlim=None, recent_window=2.5, max_files_for_scan=None):
    files = list_disk_snapshots(snapshot_dir)
    if len(files) == 0:
        raise RuntimeError(f"No snapshot files")
    if max_files_for_scan is not None and len(files) > max_files_for_scan:
        idx_scan = np.linspace(0, len(files) - 1, max_files_for_scan).astype(int)
    else:
        idx_scan = np.arange(len(files), dtype=int)
    recent_cache = RecentWindowMaxCache(files, par, window=recent_window)
    buffers = {k: [] for k in [
        "p", "rho", "u", "v", "T", "lam", "schlieren", "omega", "abs_omega",
        "pmax", "pmax_contrast", "Tmax", "p_recent", "p_recent_contrast",
        "avg_p", "avg_rho", "avg_u", "avg_v", "avg_T", "avg_lam",
    ]}
    for idx in idx_scan:
        snap = load_disk_snapshot(files[int(idx)])
        x, y, U = snap["x"], snap["y"], snap["U"]
        xlim = get_fixed_xlim(x, fixed_xlim)
        mask = x_mask_from_xlim(x, xlim)
        rho, u, v, p, T, lam = primitive_fields_from_U(U, par)
        schl = schlieren_from_fields(x, y, rho)
        omega = vorticity_from_fields(x, y, u, v)
        for name, arr in [("p", p), ("rho", rho), ("u", u), ("v", v), ("T", T), ("lam", lam), ("schlieren", schl), ("omega", omega)]:
            buffers[name].append(arr[mask, :].ravel())
        buffers["abs_omega"].append(np.abs(omega[mask, :]).ravel())
        p_max = snap["p_max"] if snap["p_max"] is not None else p
        T_max = snap["T_max"] if snap["T_max"] is not None else T
        buffers["pmax"].append(p_max[mask, :].ravel())
        buffers["pmax_contrast"].append(contrast_y(p_max)[mask, :].ravel())
        buffers["Tmax"].append(T_max[mask, :].ravel())
        p_recent = recent_cache.p_recent_for_index(int(idx))
        buffers["p_recent"].append(p_recent[mask, :].ravel())
        buffers["p_recent_contrast"].append(contrast_y(p_recent)[mask, :].ravel())
        avg = y_average_from_U(U, par)
        for key, name in [("avg_p", "p"), ("avg_rho", "rho"), ("avg_u", "u"), ("avg_v", "v"), ("avg_T", "T"), ("avg_lam", "lam")]:
            buffers[key].append(avg[name][mask])
    allvals = {k: np.concatenate(v) if len(v) else np.array([0.0]) for k, v in buffers.items()}
    return {
        "p": robust_limits(allvals["p"], 1, 99),
        "rho": robust_limits(allvals["rho"], 1, 99),
        "u": robust_limits(allvals["u"], 1, 99, symmetric=True),
        "v": robust_limits(allvals["v"], 1, 99, symmetric=True),
        "T": robust_limits(allvals["T"], 1, 99),
        "lam": (0.0, 1.0),
        "schlieren": robust_limits(allvals["schlieren"], 1, 99.5),
        "omega": robust_limits(allvals["omega"], 1, 99, symmetric=True),
        "abs_omega": robust_limits(allvals["abs_omega"], 1, 99.5),
        "pmax": robust_limits(allvals["pmax"], 1, 99),
        "pmax_contrast": robust_limits(allvals["pmax_contrast"], 1, 99, symmetric=True),
        "Tmax": robust_limits(allvals["Tmax"], 1, 99),
        "p_recent": robust_limits(allvals["p_recent"], 1, 99),
        "p_recent_contrast": robust_limits(allvals["p_recent_contrast"], 1, 99, symmetric=True),
        "avg_p": robust_limits(allvals["avg_p"], 1, 99),
        "avg_rho": robust_limits(allvals["avg_rho"], 1, 99),
        "avg_u": robust_limits(allvals["avg_u"], 1, 99, symmetric=True),
        "avg_v": robust_limits(allvals["avg_v"], 1, 99, symmetric=True),
        "avg_T": robust_limits(allvals["avg_T"], 1, 99),
        "avg_lam": (-0.02, 1.02),
    }


def _plot_two_imshow_panels(x, y, arr1, arr2, title1, title2, key1, key2, out_file, t=None, fixed_xlim=None, limits=None, cmap1=CMAP_SCALAR, cmap2=CMAP_SCALAR, cbar_label1=None, cbar_label2=None, dpi=120, figsize=FIGSIZE_PAIR):
    xlim = get_fixed_xlim(x, fixed_xlim)
    mask = x_mask_from_xlim(x, xlim)
    xplot = x[mask]
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, constrained_layout=False)
    for ax, arr, title, key, cmap, cbar_label in [
        (axes[0], arr1, title1, key1, cmap1, cbar_label1),
        (axes[1], arr2, title2, key2, cmap2, cbar_label2),
    ]:
        vmin, vmax = limits[key] if limits is not None and key in limits else (None, None)
        im = ax.imshow(arr[mask, :].T, origin="lower", aspect="auto", extent=[xplot[0], xplot[-1], y[0], y[-1]], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("y")
        cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
        if cbar_label:
            cb.set_label(cbar_label)
    axes[-1].set_xlabel("x")
    if t is not None:
        fig.suptitle(f"t = {t:.3f}", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])
    return save_fixed_size_figure(fig, out_file, dpi=dpi)


def render_yavg_full_frame(snapshot_file, par, out_dirs, frame_id, fixed_xlim=None, limits=None, dpi=120):
    snap = load_disk_snapshot(snapshot_file)
    t, x, U = snap["t"], snap["x"], snap["U"]
    avg = y_average_from_U(U, par)
    xlim = get_fixed_xlim(x, fixed_xlim)
    mask = x_mask_from_xlim(x, xlim)
    names = ["p", "rho", "u", "T", "lam"]
    labels = ["p", r"$\rho$", "u", "T", r"$\lambda$"]
    keys = ["avg_p", "avg_rho", "avg_u", "avg_T", "avg_lam"]
    fig, axes = plt.subplots(5, 1, figsize=FIGSIZE_YAVG_FULL, sharex=True, constrained_layout=False)
    for ax, name, label, key in zip(axes, names, labels, keys):
        ax.plot(x[mask], avg[name][mask], lw=1.5)
        ax.set_ylabel(label)
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.18)
        if limits is not None and key in limits:
            ax.set_ylim(*limits[key])
    axes[0].set_title(f"Independent 2D lab-frame ignition: y-averaged evolution, t = {t:.3f}")
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    out_file = os.path.join(out_dirs["yavg_profiles"], f"frame_{frame_id:05d}.png")
    return save_fixed_size_figure(fig, out_file, dpi=dpi)


def render_2d_field_pair_frames(snapshot_file, par, out_dirs, frame_id, fixed_xlim=None, limits=None, dpi=120):
    snap = load_disk_snapshot(snapshot_file)
    t, x, y, U = snap["t"], snap["x"], snap["y"], snap["U"]
    rho, u, v, p, T, lam = primitive_fields_from_U(U, par)
    outputs = []
    outputs.append(_plot_two_imshow_panels(x, y, p, rho, "p", r"$\rho$", "p", "rho", os.path.join(out_dirs["fields_p_rho"], f"frame_{frame_id:05d}.png"), t=t, fixed_xlim=fixed_xlim, limits=limits, cmap1=CMAP_SCALAR, cmap2=CMAP_SCALAR, dpi=dpi, figsize=FIGSIZE_PAIR))
    outputs.append(_plot_two_imshow_panels(x, y, u, v, "u", "v", "u", "v", os.path.join(out_dirs["fields_u_v"], f"frame_{frame_id:05d}.png"), t=t, fixed_xlim=fixed_xlim, limits=limits, cmap1=CMAP_DIVERGING, cmap2=CMAP_DIVERGING, dpi=dpi, figsize=FIGSIZE_PAIR))
    outputs.append(_plot_two_imshow_panels(x, y, T, lam, "T", r"$\lambda$", "T", "lam", os.path.join(out_dirs["fields_T_lambda"], f"frame_{frame_id:05d}.png"), t=t, fixed_xlim=fixed_xlim, limits=limits, cmap1=CMAP_T, cmap2=CMAP_SCALAR, dpi=dpi, figsize=FIGSIZE_PAIR))
    return outputs


def render_cellular_accumulated_pair_frames(snapshot_file, par, out_dirs, frame_id, fixed_xlim=None, limits=None, dpi=120):
    snap = load_disk_snapshot(snapshot_file)
    t, x, y, U = snap["t"], snap["x"], snap["y"], snap["U"]
    rho, u, v, p, T, lam = primitive_fields_from_U(U, par)
    schl = schlieren_from_fields(x, y, rho)
    p_max = snap["p_max"] if snap["p_max"] is not None else p
    T_max = snap["T_max"] if snap["T_max"] is not None else T
    p_contrast = contrast_y(p_max)
    outputs = []
    outputs.append(_plot_two_imshow_panels(x, y, p_max, p_contrast, r"accumulated $p_{\max}(x,y)$", r"accumulated $p_{\max}/\langle p_{\max}\rangle_y - 1$", "pmax", "pmax_contrast", os.path.join(out_dirs["cellular_accum_pmax_contrast"], f"frame_{frame_id:05d}.png"), t=t, fixed_xlim=fixed_xlim, limits=limits, cmap1=CMAP_SCALAR, cmap2=CMAP_DIVERGING, dpi=dpi, figsize=FIGSIZE_PAIR))
    outputs.append(_plot_two_imshow_panels(x, y, T_max, schl, r"accumulated $T_{\max}(x,y)$", r"$\log(1+|\nabla \rho|)$", "Tmax", "schlieren", os.path.join(out_dirs["cellular_accum_Tmax_schlieren"], f"frame_{frame_id:05d}.png"), t=t, fixed_xlim=fixed_xlim, limits=limits, cmap1=CMAP_T, cmap2=CMAP_SCHLIEREN, dpi=dpi, figsize=FIGSIZE_PAIR))
    return outputs


def render_cellular_recent_pair_frames(snapshot_file, files, recent_cache, par, out_dirs, frame_id, fixed_xlim=None, limits=None, dpi=120):
    snap = load_disk_snapshot(snapshot_file)
    t, x, y = snap["t"], snap["x"], snap["y"]
    p_recent = recent_cache.p_recent_for_index(frame_id)
    p_recent_contrast = contrast_y(p_recent)
    return [_plot_two_imshow_panels(x, y, p_recent, p_recent_contrast, rf"recent-window $p_{{\max}}(x,y)$, $\Delta t={recent_cache.window:g}$", rf"recent-window $p_{{\max}}/\langle p_{{\max}}\rangle_y - 1$, $\Delta t={recent_cache.window:g}$", "p_recent", "p_recent_contrast", os.path.join(out_dirs["cellular_recent_pmax_contrast"], f"frame_{frame_id:05d}.png"), t=t, fixed_xlim=fixed_xlim, limits=limits, cmap1=CMAP_SCALAR, cmap2=CMAP_DIVERGING, dpi=dpi, figsize=FIGSIZE_PAIR)]


def render_vorticity_pair_frames(snapshot_file, par, out_dirs, frame_id, fixed_xlim=None, limits=None, dpi=120):
    snap = load_disk_snapshot(snapshot_file)
    t, x, y, U = snap["t"], snap["x"], snap["y"], snap["U"]
    rho, u, v, p, T, lam = primitive_fields_from_U(U, par)
    omega = vorticity_from_fields(x, y, u, v)
    abs_omega = np.abs(omega)
    return [_plot_two_imshow_panels(x, y, omega, abs_omega, r"signed vorticity $\omega_z=\partial_x v-\partial_y u$", r"absolute vorticity $|\omega_z|$", "omega", "abs_omega", os.path.join(out_dirs["vorticity"], f"frame_{frame_id:05d}.png"), t=t, fixed_xlim=fixed_xlim, limits=limits, cmap1=CMAP_DIVERGING, cmap2=CMAP_SCALAR, dpi=dpi, figsize=FIGSIZE_PAIR_TALL)]


def render_fixed_lab_pair_animation_frames(snapshot_dir, par, frame_root, fixed_xlim=None, recent_window=1.5, dpi=120, clean=True, scan_all_for_limits=True):
    files = list_disk_snapshots(snapshot_dir)
    if len(files) == 0:
        raise RuntimeError(f"No snapshot_*.npz files found in {snapshot_dir}")
    if clean:
        safe_clean_dir(frame_root)
    else:
        ensure_dir(frame_root)
    video_names = [
        "yavg_profiles",
        "fields_p_rho",
        "fields_u_v",
        "fields_T_lambda",
        "cellular_accum_pmax_contrast",
        "cellular_accum_Tmax_schlieren",
        "cellular_recent_pmax_contrast",
        "vorticity",
    ]
    out_dirs = {name: ensure_dir(os.path.join(str(frame_root), name)) for name in video_names}
    max_scan = None if scan_all_for_limits else min(25, len(files))
    print("Computing stable color/y limits...")
    limits = compute_animation_limits(snapshot_dir, par, fixed_xlim=fixed_xlim, recent_window=recent_window, max_files_for_scan=max_scan)
    recent_cache = RecentWindowMaxCache(files, par, window=recent_window)
    rendered = {name: [] for name in video_names}
    print(f"Rendering fixed-lab pair frames from {len(files)} snapshots...")
    print(f"Frame root: {frame_root}")
    print(f"Fixed xlim: {fixed_xlim if fixed_xlim is not None else 'full domain'}")
    print(f"Recent-max window: {recent_window}")
    for i, fname in enumerate(files):
        render_yavg_full_frame(fname, par, out_dirs, i, fixed_xlim=fixed_xlim, limits=limits, dpi=dpi)
        render_2d_field_pair_frames(fname, par, out_dirs, i, fixed_xlim=fixed_xlim, limits=limits, dpi=dpi)
        render_cellular_accumulated_pair_frames(fname, par, out_dirs, i, fixed_xlim=fixed_xlim, limits=limits, dpi=dpi)
        render_cellular_recent_pair_frames(fname, files, recent_cache, par, out_dirs, i, fixed_xlim=fixed_xlim, limits=limits, dpi=dpi)
        render_vorticity_pair_frames(fname, par, out_dirs, i, fixed_xlim=fixed_xlim, limits=limits, dpi=dpi)
        for name in video_names:
            rendered[name].append(os.path.join(out_dirs[name], f"frame_{i:05d}.png"))
        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  rendered {i + 1}/{len(files)}")
    return rendered, limits


def _pad_image_to_multiple_of_16(img, background=255):
    h, w = img.shape[:2]
    target_h = int(math.ceil(h / 16.0) * 16)
    target_w = int(math.ceil(w / 16.0) * 16)
    if target_h == h and target_w == w:
        return img
    if img.ndim == 2:
        out = np.full((target_h, target_w), background, dtype=img.dtype)
        out[:h, :w] = img
    else:
        channels = img.shape[2]
        out = np.full((target_h, target_w, channels), background, dtype=img.dtype)
        out[:h, :w, :] = img
    return out


def _resize_to_target_if_needed(img, target_h, target_w):
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL/Pillow")
    pil = Image.fromarray(img)
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    pil = pil.resize((target_w, target_h), resample)
    return np.asarray(pil)


def make_video_from_frames_fixed(frame_dir, output_file, fps=20, quality=8):
    if not IMAGEIO_AVAILABLE:
        raise RuntimeError("imageio")
    files = sorted(glob.glob(os.path.join(str(frame_dir), "frame_*.png")))
    if len(files) == 0:
        print(f"No frame_*.png")
        return None
    ensure_dir(os.path.dirname(str(output_file)) or ".")
    first = imageio.imread(files[0])
    first = _pad_image_to_multiple_of_16(first)
    target_h, target_w = first.shape[:2]
    with imageio.get_writer(str(output_file), fps=fps, codec="libx264", quality=quality, macro_block_size=1) as writer:
        for fname in files:
            img = imageio.imread(fname)
            img = _resize_to_target_if_needed(img, target_h, target_w)
            img = _pad_image_to_multiple_of_16(img)
            writer.append_data(img)
    print(f"Saved video: {output_file}")
    return str(output_file)


def build_fixed_lab_pair_videos_from_frames(frame_root, video_dir, fps=20):
    ensure_dir(video_dir)
    for old_mp4 in glob.glob(os.path.join(str(video_dir), "*.mp4")):
        try:
            os.remove(old_mp4)
        except OSError:
            pass
    video_names = [
        "yavg_profiles",
        "fields_p_rho",
        "fields_u_v",
        "fields_T_lambda",
        "cellular_accum_pmax_contrast",
        "cellular_accum_Tmax_schlieren",
        "cellular_recent_pmax_contrast",
        "vorticity",
    ]
    videos = {}
    for name in video_names:
        frame_dir = os.path.join(str(frame_root), name)
        out_file = os.path.join(str(video_dir), f"{name}_fixed_lab.mp4")
        videos[name] = make_video_from_frames_fixed(frame_dir, out_file, fps=fps)
    return videos


def front_positions_by_level_lab(x, field, level, direction="decreasing", choose="rightmost"):
    # field shape: Nx, Ny. Returns one x-front per y-line.
    Nx, Ny = field.shape
    out = np.full(Ny, np.nan, dtype=float)
    if choose == "rightmost":
        iterator = range(Nx - 1, -1, -1)
    else:
        iterator = range(Nx)
    for j in range(Ny):
        for i in iterator:
            if field[i, j] >= level:
                out[j] = x[i]
                break
    return out


def finite_front_stats(x_front):
    arr = np.asarray(x_front, dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return {"mean": np.nan, "std": np.nan, "amp": np.nan, "valid_count": 0}
    vals = arr[mask]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "amp": float(np.max(vals) - np.min(vals)), "valid_count": int(mask.sum())}


def trajectory_from_npz_snapshots(snapshot_dir, par: Params, p_level=2.0, lambda_level=0.5):
    rows = []
    for f in list_disk_snapshots(snapshot_dir):
        snap = load_disk_snapshot(f)
        x, U = snap["x"], snap["U"]
        rho, u, v, p, T, lam = primitive_fields_from_U(U, par)
        xp = front_positions_by_level_lab(x, p, p_level)
        xl = front_positions_by_level_lab(x, lam, lambda_level)
        ps = finite_front_stats(xp)
        ls = finite_front_stats(xl)
        rows.append({
            "t": snap["t"],
            "x_shock_mean": ps["mean"], "x_shock_std": ps["std"], "x_shock_amp": ps["amp"], "x_shock_valid": ps["valid_count"],
            "x_lambda_mean": ls["mean"], "x_lambda_std": ls["std"], "x_lambda_amp": ls["amp"], "x_lambda_valid": ls["valid_count"],
        })
    return rows


def fit_front_speed(trajectory, t_min=4.0, key="x_shock_mean"):
    t = np.array([r["t"] for r in trajectory], dtype=float)
    x = np.array([r[key] for r in trajectory], dtype=float)
    m = np.isfinite(t) & np.isfinite(x) & (t >= t_min)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan
    coeff = np.polyfit(t[m], x[m], 1)
    pred = coeff[0] * t[m] + coeff[1]
    rms = float(np.sqrt(np.mean((x[m] - pred)**2)))
    return float(coeff[0]), float(coeff[1]), rms


def plot_diagnostics_csv(diag, out_dir):
    ensure_dir(out_dir)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(diag["t"], diag["dt"], marker="o", markersize=2, lw=1)
    axes[0].set_ylabel("dt")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(diag["t"], diag["p_max"], marker="o", markersize=2, lw=1)
    axes[1].set_ylabel("p_max")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(diag["t"], diag["shock_x_mean"], marker="o", markersize=2, lw=1)
    axes[2].set_ylabel("shock x mean")
    axes[2].set_xlabel("t")
    axes[2].grid(True, alpha=0.25)
    fig.suptitle("C++/OpenMP solver diagnostics")
    fig.tight_layout()
    return save_fixed_size_figure(fig, Path(out_dir) / "diagnostics_dt_pmax_shock.png", dpi=140)


def plot_final_fields(snapshot_file, par: Params, out_dir, xlim=None):
    ensure_dir(out_dir)
    snap = load_disk_snapshot(snapshot_file)
    x, y, U, t = snap["x"], snap["y"], snap["U"], snap["t"]
    rho, u, v, p, T, lam = primitive_fields_from_U(U, par)
    fields = [(p, "p", CMAP_SCALAR), (rho, r"$\rho$", CMAP_SCALAR), (u, "u", CMAP_DIVERGING), (v, "v", CMAP_DIVERGING), (T, "T", CMAP_T), (lam, r"$\lambda$", CMAP_SCALAR)]
    xlim = get_fixed_xlim(x, xlim)
    mask = x_mask_from_xlim(x, xlim)
    xp = x[mask]
    fig, axes = plt.subplots(len(fields), 1, figsize=(10, 14), sharex=True)
    for ax, (arr, title, cmap) in zip(axes, fields):
        im = ax.imshow(arr[mask, :].T, origin="lower", aspect="auto", extent=[xp[0], xp[-1], y[0], y[-1]], cmap=cmap)
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.024, pad=0.015)
    axes[-1].set_xlabel("x")
    fig.suptitle(f"Final C++/OpenMP fields, t = {t:.3f}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return save_fixed_size_figure(fig, Path(out_dir) / "final_2d_fields.png", dpi=140)


def plot_schlieren_and_sootfoil(snapshot_file, par: Params, out_dir, xlim=None):
    ensure_dir(out_dir)
    snap = load_disk_snapshot(snapshot_file)
    x, y, U, t = snap["x"], snap["y"], snap["U"], snap["t"]
    rho, u, v, p, T, lam = primitive_fields_from_U(U, par)
    schl = schlieren_from_fields(x, y, rho)
    pmax = snap["p_max"] if snap["p_max"] is not None else p
    contrast = contrast_y(pmax)
    xlim = get_fixed_xlim(x, xlim)
    return _plot_two_imshow_panels(x, y, schl, contrast, r"$\log(1+|\nabla\rho|)$", r"$p_{\max}/\langle p_{\max}\rangle_y - 1$", "schlieren", "pmax_contrast", Path(out_dir) / "schlieren_and_sootfoil_contrast.png", t=t, fixed_xlim=xlim, limits={"schlieren": robust_limits(schl, 1, 99.5), "pmax_contrast": robust_limits(contrast, 1, 99, symmetric=True)}, cmap1=CMAP_SCHLIEREN, cmap2=CMAP_DIVERGING, dpi=140)


def plot_front_trajectory(trajectory, out_dir, D_cj=6.415008470922, t_fit_min=4.0):
    ensure_dir(out_dir)
    t = np.array([r["t"] for r in trajectory])
    xs = np.array([r["x_shock_mean"] for r in trajectory])
    xl = np.array([r["x_lambda_mean"] for r in trajectory])
    Ds, bs, rms_s = fit_front_speed(trajectory, t_min=t_fit_min, key="x_shock_mean")
    Dl, bl, rms_l = fit_front_speed(trajectory, t_min=t_fit_min, key="x_lambda_mean")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, xs, "o-", ms=3, label=f"shock mean, D={Ds:.4f}")
    if np.isfinite(Ds): axes[0].plot(t, Ds*t + bs, "--", lw=1, label="fit")
    axes[0].set_ylabel("x shock")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(t, xl, "o-", ms=3, label=f"lambda mean, D={Dl:.4f}")
    if np.isfinite(Dl): axes[1].plot(t, Dl*t + bl, "--", lw=1, label="fit")
    axes[1].axline((0, xs[0] if len(xs) else 0), slope=D_cj, ls=":", lw=1.3, label=f"CJ slope {D_cj:.4f}")
    axes[1].set_ylabel("x lambda=0.5")
    axes[1].set_xlabel("t")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Front trajectory from C++/OpenMP snapshots")
    fig.tight_layout()
    out = save_fixed_size_figure(fig, Path(out_dir) / "front_trajectory.png", dpi=140)
    return out, {"D_shock": Ds, "D_lambda": Dl, "shock_rms": rms_s, "lambda_rms": rms_l}


def plot_front_kymographs(snapshot_dir, par: Params, out_dir, p_level=2.0, lambda_level=0.5):
    ensure_dir(out_dir)
    files = list_disk_snapshots(snapshot_dir)
    if not files:
        return None
    xs_list, xl_list, times = [], [], []
    for f in files:
        snap = load_disk_snapshot(f)
        rho, u, v, p, T, lam = primitive_fields_from_U(snap["U"], par)
        xs_list.append(front_positions_by_level_lab(snap["x"], p, p_level))
        xl_list.append(front_positions_by_level_lab(snap["x"], lam, lambda_level))
        times.append(snap["t"])
    Xs = np.asarray(xs_list)
    Xl = np.asarray(xl_list)
    y = load_disk_snapshot(files[0])["y"]
    times = np.asarray(times)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, arr, title in [(axes[0], Xs, "shock front x(y,t), p threshold"), (axes[1], Xl, r"reaction front x(y,t), $\lambda=0.5$")]:
        im = ax.imshow(arr.T, origin="lower", aspect="auto", extent=[times[0], times[-1], y[0], y[-1]], cmap=CMAP_SCALAR)
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    return save_fixed_size_figure(fig, Path(out_dir) / "front_kymographs.png", dpi=140)


def plot_reaction_zone_thickness(snapshot_dir, par: Params, out_dir, p_level=2.0, lambda_level=0.5):
    ensure_dir(out_dir)
    files = list_disk_snapshots(snapshot_dir)
    times, mean_thick, std_thick = [], [], []
    for f in files:
        snap = load_disk_snapshot(f)
        rho, u, v, p, T, lam = primitive_fields_from_U(snap["U"], par)
        xs = front_positions_by_level_lab(snap["x"], p, p_level)
        xl = front_positions_by_level_lab(snap["x"], lam, lambda_level)
        d = xs - xl
        m = np.isfinite(d)
        times.append(snap["t"])
        mean_thick.append(float(np.nanmean(d[m])) if np.any(m) else np.nan)
        std_thick.append(float(np.nanstd(d[m])) if np.any(m) else np.nan)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, mean_thick, "o-", ms=3, label="mean shock-lambda distance")
    ax.fill_between(times, np.asarray(mean_thick)-np.asarray(std_thick), np.asarray(mean_thick)+np.asarray(std_thick), alpha=0.2, label="± std_y")
    ax.set_xlabel("t")
    ax.set_ylabel("reaction-zone thickness")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return save_fixed_size_figure(fig, Path(out_dir) / "reaction_zone_thickness.png", dpi=140)


def plot_vorticity_field(snapshot_file, par: Params, out_dir, xlim=None):
    ensure_dir(out_dir)
    snap = load_disk_snapshot(snapshot_file)
    x, y, U, t = snap["x"], snap["y"], snap["U"], snap["t"]
    rho, u, v, p, T, lam = primitive_fields_from_U(U, par)
    omega = vorticity_from_fields(x, y, u, v)
    return _plot_two_imshow_panels(x, y, omega, np.abs(omega), r"signed vorticity $\omega_z$", r"absolute vorticity $|\omega_z|$", "omega", "abs_omega", Path(out_dir) / "vorticity_final.png", t=t, fixed_xlim=xlim, limits={"omega": robust_limits(omega, 1, 99, symmetric=True), "abs_omega": robust_limits(np.abs(omega), 1, 99.5)}, cmap1=CMAP_DIVERGING, cmap2=CMAP_SCALAR, dpi=140)
