#pragma once
#include <algorithm>
#include <cmath>
#include "state.hpp"

inline double minmod2(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

inline double minmod3(double a, double b, double c) {
    return minmod2(a, minmod2(b, c));
}

inline double mc_slope(double dm, double dp) {
    return minmod3(0.5 * (dm + dp), 2.0 * dm, 2.0 * dp);
}

inline Prim x_recon_left(const std::vector<double>& U, int i, int j, int Nx, int Ny, const Params& par, bool use_muscl) {
    Prim W = prim_from_cons_cell(U, i, j, Ny, par);
    if (!use_muscl || i == 0 || i == Nx + 1) return W;

    const Prim M = prim_from_cons_cell(U, i - 1, j, Ny, par);
    const Prim P = prim_from_cons_cell(U, i + 1, j, Ny, par);
    W.rho += 0.5 * mc_slope(W.rho - M.rho, P.rho - W.rho);
    W.u   += 0.5 * mc_slope(W.u   - M.u,   P.u   - W.u);
    W.v   += 0.5 * mc_slope(W.v   - M.v,   P.v   - W.v);
    W.p   += 0.5 * mc_slope(W.p   - M.p,   P.p   - W.p);
    W.lam += 0.5 * mc_slope(W.lam - M.lam, P.lam - W.lam);
    W.rho = std::max(W.rho, POSITIVITY_FLOOR_RHO);
    W.p   = std::max(W.p,   POSITIVITY_FLOOR_P);
    W.lam = clamp_value(W.lam, 0.0, 1.0);
    return W;
}

inline Prim x_recon_right(const std::vector<double>& U, int i, int j, int Nx, int Ny, const Params& par, bool use_muscl) {
    Prim W = prim_from_cons_cell(U, i, j, Ny, par);
    if (!use_muscl || i == 0 || i == Nx + 1) return W;

    const Prim M = prim_from_cons_cell(U, i - 1, j, Ny, par);
    const Prim P = prim_from_cons_cell(U, i + 1, j, Ny, par);
    W.rho -= 0.5 * mc_slope(W.rho - M.rho, P.rho - W.rho);
    W.u   -= 0.5 * mc_slope(W.u   - M.u,   P.u   - W.u);
    W.v   -= 0.5 * mc_slope(W.v   - M.v,   P.v   - W.v);
    W.p   -= 0.5 * mc_slope(W.p   - M.p,   P.p   - W.p);
    W.lam -= 0.5 * mc_slope(W.lam - M.lam, P.lam - W.lam);
    W.rho = std::max(W.rho, POSITIVITY_FLOOR_RHO);
    W.p   = std::max(W.p,   POSITIVITY_FLOOR_P);
    W.lam = clamp_value(W.lam, 0.0, 1.0);
    return W;
}

inline Prim y_recon_bottom(const std::vector<double>& U, int i, int j, int Ny, const Params& par, bool use_muscl) {
    Prim W = prim_from_cons_cell(U, i, j, Ny, par);
    if (!use_muscl || j == 0 || j == Ny + 1) return W;

    const Prim M = prim_from_cons_cell(U, i, j - 1, Ny, par);
    const Prim P = prim_from_cons_cell(U, i, j + 1, Ny, par);
    W.rho += 0.5 * mc_slope(W.rho - M.rho, P.rho - W.rho);
    W.u   += 0.5 * mc_slope(W.u   - M.u,   P.u   - W.u);
    W.v   += 0.5 * mc_slope(W.v   - M.v,   P.v   - W.v);
    W.p   += 0.5 * mc_slope(W.p   - M.p,   P.p   - W.p);
    W.lam += 0.5 * mc_slope(W.lam - M.lam, P.lam - W.lam);
    W.rho = std::max(W.rho, POSITIVITY_FLOOR_RHO);
    W.p   = std::max(W.p,   POSITIVITY_FLOOR_P);
    W.lam = clamp_value(W.lam, 0.0, 1.0);
    return W;
}

inline Prim y_recon_top(const std::vector<double>& U, int i, int j, int Ny, const Params& par, bool use_muscl) {
    Prim W = prim_from_cons_cell(U, i, j, Ny, par);
    if (!use_muscl || j == 0 || j == Ny + 1) return W;

    const Prim M = prim_from_cons_cell(U, i, j - 1, Ny, par);
    const Prim P = prim_from_cons_cell(U, i, j + 1, Ny, par);
    W.rho -= 0.5 * mc_slope(W.rho - M.rho, P.rho - W.rho);
    W.u   -= 0.5 * mc_slope(W.u   - M.u,   P.u   - W.u);
    W.v   -= 0.5 * mc_slope(W.v   - M.v,   P.v   - W.v);
    W.p   -= 0.5 * mc_slope(W.p   - M.p,   P.p   - W.p);
    W.lam -= 0.5 * mc_slope(W.lam - M.lam, P.lam - W.lam);
    W.rho = std::max(W.rho, POSITIVITY_FLOOR_RHO);
    W.p   = std::max(W.p,   POSITIVITY_FLOOR_P);
    W.lam = clamp_value(W.lam, 0.0, 1.0);
    return W;
}

inline Flux hllc_flux_x(const Prim& L, const Prim& R, const Params& par) {
    const double gamma = par.gamma;
    const double EL = energy_from_prim(L, par);
    const double ER = energy_from_prim(R, par);
    const Cons UL{L.rho, L.rho * L.u, L.rho * L.v, EL, L.rho * L.lam};
    const Cons UR{R.rho, R.rho * R.u, R.rho * R.v, ER, R.rho * R.lam};

    const Flux FL{UL[MX], UL[MX] * L.u + L.p, UL[MY] * L.u, L.u * (EL + L.p), L.u * UL[RHL]};
    const Flux FR{UR[MX], UR[MX] * R.u + R.p, UR[MY] * R.u, R.u * (ER + R.p), R.u * UR[RHL]};

    const double cL = std::sqrt(gamma * L.p / L.rho);
    const double cR = std::sqrt(gamma * R.p / R.rho);
    const double SL = std::min(L.u - cL, R.u - cR);
    const double SR = std::max(L.u + cL, R.u + cR);
    const double denom = safe_den(L.rho * (SL - L.u) - R.rho * (SR - R.u));
    const double SM = (R.p - L.p + L.rho * L.u * (SL - L.u) - R.rho * R.u * (SR - R.u)) / denom;
    const double pstarL = L.p + L.rho * (SL - L.u) * (SM - L.u);
    const double pstarR = R.p + R.rho * (SR - R.u) * (SM - R.u);
    const double pstar = std::max(0.5 * (pstarL + pstarR), POSITIVITY_FLOOR_P);

    if (0.0 <= SL) return FL;

    Flux out{};
    if (SL <= 0.0 && 0.0 <= SM) {
        const double den = safe_den(SL - SM);
        const double rS = std::max(L.rho * (SL - L.u) / den, POSITIVITY_FLOOR_RHO);
        const Cons US{rS, rS * SM, rS * L.v,
                      ((SL - L.u) * EL - L.p * L.u + pstar * SM) / den,
                      rS * L.lam};
        for (int m = 0; m < NV; ++m) out[m] = FL[m] + SL * (US[m] - UL[m]);
        return out;
    }

    if (SM <= 0.0 && 0.0 <= SR) {
        const double den = safe_den(SR - SM);
        const double rS = std::max(R.rho * (SR - R.u) / den, POSITIVITY_FLOOR_RHO);
        const Cons US{rS, rS * SM, rS * R.v,
                      ((SR - R.u) * ER - R.p * R.u + pstar * SM) / den,
                      rS * R.lam};
        for (int m = 0; m < NV; ++m) out[m] = FR[m] + SR * (US[m] - UR[m]);
        return out;
    }

    return FR;
}

inline Flux hllc_flux_y(const Prim& B, const Prim& T, const Params& par) {
    const double gamma = par.gamma;
    const double EB = energy_from_prim(B, par);
    const double ET = energy_from_prim(T, par);
    const Cons UB{B.rho, B.rho * B.u, B.rho * B.v, EB, B.rho * B.lam};
    const Cons UT{T.rho, T.rho * T.u, T.rho * T.v, ET, T.rho * T.lam};

    const Flux GB{UB[MY], UB[MX] * B.v, UB[MY] * B.v + B.p, B.v * (EB + B.p), B.v * UB[RHL]};
    const Flux GT{UT[MY], UT[MX] * T.v, UT[MY] * T.v + T.p, T.v * (ET + T.p), T.v * UT[RHL]};

    const double cB = std::sqrt(gamma * B.p / B.rho);
    const double cT = std::sqrt(gamma * T.p / T.rho);
    const double SL = std::min(B.v - cB, T.v - cT);
    const double SR = std::max(B.v + cB, T.v + cT);
    const double denom = safe_den(B.rho * (SL - B.v) - T.rho * (SR - T.v));
    const double SM = (T.p - B.p + B.rho * B.v * (SL - B.v) - T.rho * T.v * (SR - T.v)) / denom;
    const double pstarB = B.p + B.rho * (SL - B.v) * (SM - B.v);
    const double pstarT = T.p + T.rho * (SR - T.v) * (SM - T.v);
    const double pstar = std::max(0.5 * (pstarB + pstarT), POSITIVITY_FLOOR_P);

    if (0.0 <= SL) return GB;

    Flux out{};
    if (SL <= 0.0 && 0.0 <= SM) {
        const double den = safe_den(SL - SM);
        const double rS = std::max(B.rho * (SL - B.v) / den, POSITIVITY_FLOOR_RHO);
        const Cons US{rS, rS * B.u, rS * SM,
                      ((SL - B.v) * EB - B.p * B.v + pstar * SM) / den,
                      rS * B.lam};
        for (int m = 0; m < NV; ++m) out[m] = GB[m] + SL * (US[m] - UB[m]);
        return out;
    }

    if (SM <= 0.0 && 0.0 <= SR) {
        const double den = safe_den(SR - SM);
        const double rS = std::max(T.rho * (SR - T.v) / den, POSITIVITY_FLOOR_RHO);
        const Cons US{rS, rS * T.u, rS * SM,
                      ((SR - T.v) * ET - T.p * T.v + pstar * SM) / den,
                      rS * T.lam};
        for (int m = 0; m < NV; ++m) out[m] = GT[m] + SR * (US[m] - UT[m]);
        return out;
    }

    return GT;
}
