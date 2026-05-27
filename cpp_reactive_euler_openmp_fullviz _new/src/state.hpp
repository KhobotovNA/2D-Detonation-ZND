#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "params.hpp"

constexpr int NV = 5;
constexpr int RHO = 0;
constexpr int MX  = 1;
constexpr int MY  = 2;
constexpr int ENE = 3;
constexpr int RHL = 4;

constexpr double POSITIVITY_FLOOR_RHO = 1.0e-10;
constexpr double POSITIVITY_FLOOR_P   = 1.0e-10;
constexpr double SMALL_DENOM          = 1.0e-14;

using Cons = std::array<double, NV>;
using Flux = std::array<double, NV>;

struct Prim {
    double rho = 1.0;
    double u   = 0.0;
    double v   = 0.0;
    double p   = 1.0;
    double lam = 0.0;
};

inline double clamp_value(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

inline double safe_den(double x) {
    if (std::abs(x) < SMALL_DENOM) return (x >= 0.0 ? SMALL_DENOM : -SMALL_DENOM);
    return x;
}

inline std::size_t idx(int i, int j, int m, int Ny) {
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(Ny + 2) + static_cast<std::size_t>(j)) * NV + static_cast<std::size_t>(m);
}

inline double energy_from_prim(const Prim& W, const Params& par) {
    const double rho = std::max(W.rho, POSITIVITY_FLOOR_RHO);
    const double p   = std::max(W.p,   POSITIVITY_FLOOR_P);
    const double lam = clamp_value(W.lam, 0.0, 1.0);
    return p / (par.gamma - 1.0) + 0.5 * rho * (W.u * W.u + W.v * W.v) - par.Q * rho * lam;
}

inline Cons cons_from_prim(const Prim& W_in, const Params& par) {
    Prim W = W_in;
    W.rho = std::max(W.rho, POSITIVITY_FLOOR_RHO);
    W.p   = std::max(W.p,   POSITIVITY_FLOOR_P);
    W.lam = clamp_value(W.lam, 0.0, 1.0);
    const double E = energy_from_prim(W, par);
    return Cons{W.rho, W.rho * W.u, W.rho * W.v, E, W.rho * W.lam};
}

inline Prim prim_from_cons_cell(const std::vector<double>& U, int i, int j, int Ny, const Params& par) {
    Prim W;
    W.rho = std::max(U[idx(i, j, RHO, Ny)], POSITIVITY_FLOOR_RHO);
    const double mx = U[idx(i, j, MX,  Ny)];
    const double my = U[idx(i, j, MY,  Ny)];
    const double E  = U[idx(i, j, ENE, Ny)];
    const double z  = U[idx(i, j, RHL, Ny)];

    W.u = mx / W.rho;
    W.v = my / W.rho;
    W.lam = clamp_value(z / W.rho, 0.0, 1.0);

    const double kinetic = 0.5 * (mx * mx + my * my) / W.rho;
    W.p = (par.gamma - 1.0) * (E - kinetic + par.Q * z);
    W.p = std::max(W.p, POSITIVITY_FLOOR_P);
    return W;
}

inline void write_cons_cell(std::vector<double>& U, int i, int j, int Ny, const Cons& C) {
    for (int m = 0; m < NV; ++m) U[idx(i, j, m, Ny)] = C[m];
}

inline void sanitize_cell(std::vector<double>& U, int i, int j, int Ny, const Params& par) {
    Prim W = prim_from_cons_cell(U, i, j, Ny, par);
    write_cons_cell(U, i, j, Ny, cons_from_prim(W, par));
}
