import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NV = 5
MAGIC = 20260527


HEADER_FMT = "iii7d"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def read_snapshot(path):
    path = Path(path)
    with path.open("rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        magic, Nx, Ny, t, xL, xR, yL, yR, dx, dy = struct.unpack(HEADER_FMT, header_bytes)
        if magic != MAGIC:
            raise ValueError(f"Bad m")
        data = np.fromfile(f, dtype=np.float64)
    U = data.reshape((Nx, Ny, NV))
    x = xL + (np.arange(Nx) + 0.5) * dx
    y = yL + (np.arange(Ny) + 0.5) * dy
    return {"U": U, "x": x, "y": y, "t": t, "dx": dx, "dy": dy}


def primitive_fields(U, gamma=1.4, Q=20.0):
    rho = np.maximum(U[..., 0], 1.0e-10)
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    z = U[..., 4]
    lam = np.clip(z / rho, 0.0, 1.0)
    kinetic = 0.5 * (U[..., 1] ** 2 + U[..., 2] ** 2) / rho
    p = (gamma - 1.0) * (U[..., 3] - kinetic + Q * z)
    p = np.maximum(p, 1.0e-10)
    return rho, u, v, p, lam


def plot_snapshot(path, gamma=1.4, Q=20.0, xlim=None):
    snap = read_snapshot(path)
    U, x, y, t = snap["U"], snap["x"], snap["y"], snap["t"]
    rho, u, v, p, lam = primitive_fields(U, gamma=gamma, Q=Q)

    if xlim is not None:
        mask = (x >= xlim[0]) & (x <= xlim[1])
        x = x[mask]
        p = p[mask, :]
        lam = lam[mask, :]

    extent = [x[0], x[-1], y[0], y[-1]]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(p.T, origin="lower", extent=extent, aspect="auto")
    ax.set_title(f"Pressure, t={t:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(lam.T, origin="lower", extent=extent, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(f"Lambda, t={t:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, p.mean(axis=1), label="mean p")
    ax.plot(x, lam.mean(axis=1), label="mean lambda")
    ax.set_title(f"Y-averaged profiles, t={t:.4f}")
    ax.set_xlabel("x")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", help="Path to snapshot_XXXXX.bin")
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    args = parser.parse_args()
    plot_snapshot(args.snapshot, xlim=args.xlim)
