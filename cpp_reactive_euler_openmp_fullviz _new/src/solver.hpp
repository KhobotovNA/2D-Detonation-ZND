#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#include "flux.hpp"
#include "io.hpp"
#include "params.hpp"
#include "state.hpp"

class Solver2D {
public:
    Solver2D(Params par, RunConfig cfg, IgnitionConfig ign)
        : par_(par), cfg_(cfg), ign_(ign) {
        dx_ = (cfg_.xR - cfg_.xL) / static_cast<double>(cfg_.Nx);
        dy_ = (cfg_.yR - cfg_.yL) / static_cast<double>(cfg_.Ny);
        const std::size_t n_cells = static_cast<std::size_t>(cfg_.Nx + 2) * static_cast<std::size_t>(cfg_.Ny + 2) * NV;
        U_.assign(n_cells, 0.0);
        U0_.assign(n_cells, 0.0);
        U1_.assign(n_cells, 0.0);
        U2_.assign(n_cells, 0.0);
        K_.assign(n_cells, 0.0);
        Fhat_.assign(static_cast<std::size_t>(cfg_.Nx + 1) * static_cast<std::size_t>(cfg_.Ny) * NV, 0.0);
        Ghat_.assign(static_cast<std::size_t>(cfg_.Nx) * static_cast<std::size_t>(cfg_.Ny + 1) * NV, 0.0);
        pmax_hist_.assign(static_cast<std::size_t>(cfg_.Nx) * static_cast<std::size_t>(cfg_.Ny), par_.p_a);
        Tmax_hist_.assign(static_cast<std::size_t>(cfg_.Nx) * static_cast<std::size_t>(cfg_.Ny), par_.p_a / (par_.rho_a * par_.Rgas));
    }

    void initialize() {
        const double Ly = cfg_.yR - cfg_.yL;
#pragma omp parallel for schedule(static)
        for (int i = 1; i <= cfg_.Nx; ++i) {
            const double x = cfg_.xL + (static_cast<double>(i - 1) + 0.5) * dx_;
            for (int j = 1; j <= cfg_.Ny; ++j) {
                const double y = cfg_.yL + (static_cast<double>(j - 1) + 0.5) * dy_;
                Prim W{par_.rho_a, par_.u_a, par_.v_a, par_.p_a, par_.lam_a};

                const double x_right = ign_.x_ign_right + fourier_shift_y(y, Ly);
                if (ign_.smooth) {
                    const double sw = std::max(ign_.smooth_width, 1.0e-12);
                    const double left_w  = 0.5 * (1.0 + std::tanh((x - ign_.x_ign_left) / sw));
                    const double right_w = 0.5 * (1.0 - std::tanh((x - x_right) / sw));
                    const double w = left_w * right_w;
                    W.rho += w * (ign_.rho_ign - W.rho);
                    W.u   += w * (ign_.u_ign   - W.u);
                    W.v   += w * (ign_.v_ign   - W.v);
                    W.p   += w * (ign_.p_ign   - W.p);
                    W.lam += w * (ign_.lam_ign - W.lam);
                } else if (x >= ign_.x_ign_left && x <= x_right) {
                    W.rho = ign_.rho_ign;
                    W.u   = ign_.u_ign;
                    W.v   = ign_.v_ign;
                    W.p   = ign_.p_ign;
                    W.lam = ign_.lam_ign;
                }

                const double theta = 2.0 * M_PI * static_cast<double>(ign_.seed_mode) * (y - cfg_.yL) / Ly + ign_.seed_phase;
                const double wx = (ign_.seed_sigma_x > 0.0)
                    ? std::exp(-0.5 * std::pow((x - ign_.seed_center_x) / ign_.seed_sigma_x, 2.0))
                    : 1.0;
                if (ign_.p_amp != 0.0) {
                    W.p *= std::max(1.0 + ign_.p_amp * std::sin(theta) * wx, 1.0e-8);
                }
                if (ign_.v_amp != 0.0) {
                    W.v += ign_.v_amp * std::sin(theta + ign_.seed_phase_v) * wx;
                }

                for (const auto& hs : ign_.hotspots) {
                    const double dyper = periodic_distance(y, hs.center_y, Ly);
                    const double gx = (hs.sigma_x > 0.0)
                        ? std::exp(-0.5 * std::pow((x - hs.center_x) / hs.sigma_x, 2.0))
                        : 1.0;
                    const double gy = (hs.sigma_y > 0.0)
                        ? std::exp(-0.5 * std::pow(dyper / hs.sigma_y, 2.0))
                        : 1.0;
                    const double g = gx * gy;
                    if (hs.p_amp != 0.0) W.p *= std::max(1.0 + hs.p_amp * g, 1.0e-8);
                    if (hs.v_amp != 0.0) W.v += hs.v_amp * (dyper / std::max(hs.sigma_y, 1.0e-12)) * g;
                    if (hs.lam_amp != 0.0) W.lam += hs.lam_amp * g;
                }

                W.rho = std::max(W.rho, POSITIVITY_FLOOR_RHO);
                W.p   = std::max(W.p,   POSITIVITY_FLOOR_P);
                W.lam = clamp_value(W.lam, 0.0, 1.0);
                write_cons_cell(U_, i, j, cfg_.Ny, cons_from_prim(W, par_));
            }
        }
        apply_bc(U_);
        update_extreme_histories();
    }

    void run() {
        using clock = std::chrono::steady_clock;
        const auto wall_start = clock::now();

        ensure_directory(cfg_.output_dir);
        ensure_directory(cfg_.output_dir + "/snapshots");
        write_run_metadata(cfg_.output_dir + "/metadata.txt", cfg_, par_);
        const std::string diag_file = cfg_.output_dir + "/diagnostics.csv";
        append_diagnostics_header(diag_file);

        initialize();

        double t = 0.0;
        int step = 0;
        int snap_id = 0;
        double next_save = 0.0;
        double next_track = 0.0;

        save_snapshot(snap_id++, t);
        record_diagnostics(diag_file, step, t, 0.0);
        next_save += cfg_.save_dt;
        next_track += cfg_.track_dt;

        while (t < cfg_.final_time - 1.0e-14) {
            double dt = compute_dt();
            if (t + dt > cfg_.final_time) dt = cfg_.final_time - t;
            if (cfg_.save_snapshots && t < next_save && next_save <= t + dt) dt = next_save - t;
            if (t < next_track && next_track <= t + dt) dt = next_track - t;

            strang_step(dt);
            t += dt;
            ++step;
            update_extreme_histories();

            if (cfg_.save_snapshots && t >= next_save - 1.0e-12) {
                save_snapshot(snap_id++, t);
                next_save += cfg_.save_dt;
            }
            if (t >= next_track - 1.0e-12) {
                record_diagnostics(diag_file, step, t, dt);
                next_track += cfg_.track_dt;
            }

            if (step % 20 == 0 || t >= cfg_.final_time - 1.0e-14) {
                std::cout << "step=" << std::setw(7) << step
                          << " t=" << std::setw(10) << std::setprecision(6) << std::fixed << t
                          << " dt=" << std::scientific << dt
                          << " pmax=" << std::fixed << std::setprecision(4) << max_pressure()
                          << " shock_x=" << std::setprecision(4) << shock_x_mean(2.0)
                          << "\n";
            }
        }

        save_snapshot(snap_id++, t);
        const auto wall_end = clock::now();
        const double wall = std::chrono::duration<double>(wall_end - wall_start).count();
        std::cout << "\nFinished solver with snapshots\n";
        std::cout << "steps              = " << step << "\n";
        std::cout << "final time         = " << std::setprecision(12) << t << "\n";
        std::cout << "wall time [s]      = " << std::setprecision(6) << wall << "\n";
        std::cout << "output directory   = " << cfg_.output_dir << "\n";
        print_timing_summary(wall, step);
    }

private:
    Params par_;
    RunConfig cfg_;
    IgnitionConfig ign_;
    double dx_ = 0.0;
    double dy_ = 0.0;

    std::vector<double> U_, U0_, U1_, U2_, K_;
    std::vector<double> Fhat_, Ghat_;
    std::vector<double> pmax_hist_, Tmax_hist_;

    double time_compute_dt_ = 0.0;
    double time_reaction_   = 0.0;
    double time_rhs_        = 0.0;
    double time_rk_update_  = 0.0;
    double time_sanitize_   = 0.0;
    double time_extreme_hist_ = 0.0;
    double time_snapshot_   = 0.0;

    long long rhs_calls_ = 0;
    long long reaction_substeps_ = 0;

    static double wall_seconds_now() {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
    }

    double fourier_shift_y(double y, double Ly) const {
        double eta = ign_.edge_amp * std::sin(2.0 * M_PI * static_cast<double>(ign_.edge_mode) * (y - cfg_.yL) / Ly + ign_.edge_phase);
        for (const auto& em : ign_.edge_extra_modes) {
            eta += em.amp * std::sin(2.0 * M_PI * static_cast<double>(em.mode) * (y - cfg_.yL) / Ly + em.phase);
        }
        return eta;
    }

    static double periodic_distance(double y, double y0, double Ly) {
        double d = y - y0;
        d -= Ly * std::round(d / Ly);
        return d;
    }

    void apply_bc(std::vector<double>& U) const {

#pragma omp parallel for schedule(static)
        for (int j = 0; j < cfg_.Ny + 2; ++j) {
            for (int m = 0; m < NV; ++m) {
                U[idx(0, j, m, cfg_.Ny)] = U[idx(1, j, m, cfg_.Ny)];
                U[idx(cfg_.Nx + 1, j, m, cfg_.Ny)] = U[idx(cfg_.Nx, j, m, cfg_.Ny)];
            }
        }

 
#pragma omp parallel for schedule(static)
        for (int i = 0; i < cfg_.Nx + 2; ++i) {
            for (int m = 0; m < NV; ++m) {
                U[idx(i, 0, m, cfg_.Ny)] = U[idx(i, cfg_.Ny, m, cfg_.Ny)];
                U[idx(i, cfg_.Ny + 1, m, cfg_.Ny)] = U[idx(i, 1, m, cfg_.Ny)];
            }
        }
    }

    double compute_dt() {
        const double tic = wall_seconds_now();
        double max_x = 0.0;
        double max_y = 0.0;

#pragma omp parallel for collapse(2) schedule(static) reduction(max:max_x,max_y)
        for (int i = 1; i <= cfg_.Nx; ++i) {
            for (int j = 1; j <= cfg_.Ny; ++j) {
                const Prim W = prim_from_cons_cell(U_, i, j, cfg_.Ny, par_);
                const double c = std::sqrt(par_.gamma * W.p / W.rho);
                max_x = std::max(max_x, std::abs(W.u) + c);
                max_y = std::max(max_y, std::abs(W.v) + c);
            }
        }
        const double dt_x = dx_ / std::max(max_x, 1.0e-14);
        const double dt_y = dy_ / std::max(max_y, 1.0e-14);
        time_compute_dt_ += wall_seconds_now() - tic;
        return cfg_.cfl * std::min(dt_x, dt_y);
    }

    double omega_max() const {
        double om = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(max:om)
        for (int i = 1; i <= cfg_.Nx; ++i) {
            for (int j = 1; j <= cfg_.Ny; ++j) {
                const Prim W = prim_from_cons_cell(U_, i, j, cfg_.Ny, par_);
                const double T = std::max(W.p / (W.rho * par_.Rgas), 1.0e-14);
                const double omega = par_.k * (1.0 - W.lam) * std::exp(-par_.Ea / (par_.Rgas * T));
                om = std::max(om, omega);
            }
        }
        return om;
    }

    void reaction_update(double dtau) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i <= cfg_.Nx; ++i) {
            for (int j = 1; j <= cfg_.Ny; ++j) {
                Prim W = prim_from_cons_cell(U_, i, j, cfg_.Ny, par_);
                const double T = std::max(W.p / (W.rho * par_.Rgas), 1.0e-14);
                const double omega = par_.k * (1.0 - W.lam) * std::exp(-par_.Ea / (par_.Rgas * T));
                W.lam = clamp_value(W.lam + dtau * omega, 0.0, 1.0);
                write_cons_cell(U_, i, j, cfg_.Ny, cons_from_prim(W, par_));
            }
        }
        apply_bc(U_);
    }

    void reaction_step(double dt) {
        const double tic = wall_seconds_now();
        double remaining = dt;
        int nsub = 0;
        while (remaining > 1.0e-15) {
            const double om = omega_max();
            if (om < 1.0e-14) break;
            const double dtau = std::min(remaining, cfg_.react_cfl / om);
            reaction_update(dtau);
            remaining -= dtau;
            ++nsub;
            ++reaction_substeps_;
            if (nsub > 20000) throw std::runtime_error(" reaction steps");
        }
        time_reaction_ += wall_seconds_now() - tic;
    }

    void compute_rhs(const std::vector<double>& Ustate, std::vector<double>& rhs) {
        const double tic = wall_seconds_now();
        ++rhs_calls_;

        std::fill(rhs.begin(), rhs.end(), 0.0);

#pragma omp parallel for collapse(2) schedule(static)
        for (int ii = 0; ii <= cfg_.Nx; ++ii) {
            for (int jj0 = 0; jj0 < cfg_.Ny; ++jj0) {
                const int j = jj0 + 1;
                const Prim L = x_recon_left(Ustate, ii, j, cfg_.Nx, cfg_.Ny, par_, cfg_.use_muscl);
                const Prim R = x_recon_right(Ustate, ii + 1, j, cfg_.Nx, cfg_.Ny, par_, cfg_.use_muscl);
                const Flux F = hllc_flux_x(L, R, par_);
                for (int m = 0; m < NV; ++m) Fhat_[(static_cast<std::size_t>(ii) * cfg_.Ny + jj0) * NV + m] = F[m];
            }
        }

#pragma omp parallel for collapse(2) schedule(static)
        for (int ii0 = 0; ii0 < cfg_.Nx; ++ii0) {
            for (int jj = 0; jj <= cfg_.Ny; ++jj) {
                const int i = ii0 + 1;
                const Prim B = y_recon_bottom(Ustate, i, jj, cfg_.Ny, par_, cfg_.use_muscl);
                const Prim T = y_recon_top(Ustate, i, jj + 1, cfg_.Ny, par_, cfg_.use_muscl);
                const Flux G = hllc_flux_y(B, T, par_);
                for (int m = 0; m < NV; ++m) Ghat_[(static_cast<std::size_t>(ii0) * (cfg_.Ny + 1) + jj) * NV + m] = G[m];
            }
        }

#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i <= cfg_.Nx; ++i) {
            for (int j = 1; j <= cfg_.Ny; ++j) {
                const int ii0 = i - 1;
                const int jj0 = j - 1;
                for (int m = 0; m < NV; ++m) {
                    const double FxR = Fhat_[(static_cast<std::size_t>(ii0 + 1) * cfg_.Ny + jj0) * NV + m];
                    const double FxL = Fhat_[(static_cast<std::size_t>(ii0)     * cfg_.Ny + jj0) * NV + m];
                    const double GyT = Ghat_[(static_cast<std::size_t>(ii0) * (cfg_.Ny + 1) + jj0 + 1) * NV + m];
                    const double GyB = Ghat_[(static_cast<std::size_t>(ii0) * (cfg_.Ny + 1) + jj0)     * NV + m];
                    rhs[idx(i, j, m, cfg_.Ny)] = - (FxR - FxL) / dx_ - (GyT - GyB) / dy_;
                }
            }
        }

        time_rhs_ += wall_seconds_now() - tic;
    }

    void sanitize_inner(std::vector<double>& U) {
        const double tic = wall_seconds_now();
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i <= cfg_.Nx; ++i) {
            for (int j = 1; j <= cfg_.Ny; ++j) {
                sanitize_cell(U, i, j, cfg_.Ny, par_);
            }
        }
        apply_bc(U);
        time_sanitize_ += wall_seconds_now() - tic;
    }

    void ssprk3_step(double dt) {
        U0_ = U_;
        apply_bc(U0_);

        compute_rhs(U0_, K_);
        U1_ = U0_;
        {
            const double tic = wall_seconds_now();
#pragma omp parallel for collapse(2) schedule(static)
            for (int i = 1; i <= cfg_.Nx; ++i) {
                for (int j = 1; j <= cfg_.Ny; ++j) {
                    for (int m = 0; m < NV; ++m) {
                        const std::size_t q = idx(i, j, m, cfg_.Ny);
                        U1_[q] = U0_[q] + dt * K_[q];
                    }
                }
            }
            time_rk_update_ += wall_seconds_now() - tic;
        }
        sanitize_inner(U1_);

        compute_rhs(U1_, K_);
        U2_ = U1_;
        {
            const double tic = wall_seconds_now();
#pragma omp parallel for collapse(2) schedule(static)
            for (int i = 1; i <= cfg_.Nx; ++i) {
                for (int j = 1; j <= cfg_.Ny; ++j) {
                    for (int m = 0; m < NV; ++m) {
                        const std::size_t q = idx(i, j, m, cfg_.Ny);
                        U2_[q] = 0.75 * U0_[q] + 0.25 * (U1_[q] + dt * K_[q]);
                    }
                }
            }
            time_rk_update_ += wall_seconds_now() - tic;
        }
        sanitize_inner(U2_);

        compute_rhs(U2_, K_);
        {
            const double tic = wall_seconds_now();
#pragma omp parallel for collapse(2) schedule(static)
            for (int i = 1; i <= cfg_.Nx; ++i) {
                for (int j = 1; j <= cfg_.Ny; ++j) {
                    for (int m = 0; m < NV; ++m) {
                        const std::size_t q = idx(i, j, m, cfg_.Ny);
                        U_[q] = (1.0 / 3.0) * U0_[q] + (2.0 / 3.0) * (U2_[q] + dt * K_[q]);
                    }
                }
            }
            time_rk_update_ += wall_seconds_now() - tic;
        }
        sanitize_inner(U_);
    }

    void strang_step(double dt) {
        reaction_step(0.5 * dt);
        ssprk3_step(dt);
        reaction_step(0.5 * dt);
    }

    double max_pressure() const {
        double val = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(max:val)
        for (int i = 1; i <= cfg_.Nx; ++i) {
            for (int j = 1; j <= cfg_.Ny; ++j) {
                val = std::max(val, prim_from_cons_cell(U_, i, j, cfg_.Ny, par_).p);
            }
        }
        return val;
    }

    double max_lambda() const {
        double val = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(max:val)
        for (int i = 1; i <= cfg_.Nx; ++i) {
            for (int j = 1; j <= cfg_.Ny; ++j) {
                val = std::max(val, prim_from_cons_cell(U_, i, j, cfg_.Ny, par_).lam);
            }
        }
        return val;
    }

    void update_extreme_histories() {
        const double tic = wall_seconds_now();
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i <= cfg_.Nx; ++i) {
            for (int j = 1; j <= cfg_.Ny; ++j) {
                const Prim W = prim_from_cons_cell(U_, i, j, cfg_.Ny, par_);
                const double T = W.p / std::max(W.rho * par_.Rgas, 1.0e-14);
                const std::size_t q = static_cast<std::size_t>(i - 1) * cfg_.Ny + (j - 1);
                pmax_hist_[q] = std::max(pmax_hist_[q], W.p);
                Tmax_hist_[q] = std::max(Tmax_hist_[q], T);
            }
        }
        time_extreme_hist_ += wall_seconds_now() - tic;
    }

    double shock_x_mean(double p_level) const {
        double sum = 0.0;
        int count = 0;
#pragma omp parallel for schedule(static) reduction(+:sum,count)
        for (int j = 1; j <= cfg_.Ny; ++j) {
            double xfront = std::numeric_limits<double>::quiet_NaN();
            for (int i = cfg_.Nx; i >= 1; --i) {
                const Prim W = prim_from_cons_cell(U_, i, j, cfg_.Ny, par_);
                if (W.p >= p_level) {
                    xfront = cfg_.xL + (static_cast<double>(i - 1) + 0.5) * dx_;
                    break;
                }
            }
            if (std::isfinite(xfront)) {
                sum += xfront;
                ++count;
            }
        }
        return count > 0 ? sum / static_cast<double>(count) : std::numeric_limits<double>::quiet_NaN();
    }

    void print_timing_summary(double total_wall, int steps) const {
        const double timed_parallel = time_compute_dt_ + time_reaction_ + time_rhs_ +
                                      time_rk_update_ + time_sanitize_ + time_extreme_hist_;
        const double cell_steps_m = (steps > 0)
            ? (static_cast<double>(steps) * static_cast<double>(cfg_.Nx) * static_cast<double>(cfg_.Ny) / 1.0e6)
            : 0.0;
        const double mcell_steps_per_s = (total_wall > 0.0) ? cell_steps_m / total_wall : 0.0;

        std::cout << "\nTiming breakdown [s]\n";
        std::cout << "  compute_dt        = " << std::setprecision(6) << time_compute_dt_ << "\n";
        std::cout << "  reaction          = " << std::setprecision(6) << time_reaction_
                  << "  (substeps=" << reaction_substeps_ << ")\n";
        std::cout << "  HLLC/MUSCL RHS    = " << std::setprecision(6) << time_rhs_
                  << "  (calls=" << rhs_calls_ << ")\n";
        std::cout << "  RK vector updates = " << std::setprecision(6) << time_rk_update_ << "\n";
        std::cout << "  sanitize + BC     = " << std::setprecision(6) << time_sanitize_ << "\n";
        std::cout << "  p/T extrema hist  = " << std::setprecision(6) << time_extreme_hist_ << "\n";
        std::cout << "  snapshot I/O      = " << std::setprecision(6) << time_snapshot_ << "\n";
        std::cout << "  timed core total  = " << std::setprecision(6) << timed_parallel << "\n";
        std::cout << "  nominal MCell-step/s = " << std::setprecision(6) << mcell_steps_per_s << "\n";
    }

    void record_diagnostics(const std::string& diag_file, int step, double t, double dt) const {
        append_diagnostics_row(diag_file, step, t, dt, max_pressure(), max_lambda(), shock_x_mean(2.0));
    }

    void save_snapshot(int snap_id, double t) {
        if (!cfg_.save_snapshots) return;
        const double tic = wall_seconds_now();
        std::ostringstream name;
        name << cfg_.output_dir << "/snapshots/snapshot_" << std::setw(5) << std::setfill('0') << snap_id << ".bin";
        write_snapshot_bin(name.str(), U_, pmax_hist_, Tmax_hist_,
                           cfg_.Nx, cfg_.Ny, t, cfg_.xL, cfg_.xR, cfg_.yL, cfg_.yR, dx_, dy_);
        time_snapshot_ += wall_seconds_now() - tic;
    }
};
