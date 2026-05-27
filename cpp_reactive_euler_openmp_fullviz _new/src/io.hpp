#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "state.hpp"

struct SnapshotHeader {
    std::int32_t magic = 20260527;
    std::int32_t Nx = 0;
    std::int32_t Ny = 0;
    double t = 0.0;
    double xL = 0.0, xR = 0.0, yL = 0.0, yR = 0.0;
    double dx = 0.0, dy = 0.0;
};

inline void ensure_directory(const std::string& path) {
    std::filesystem::create_directories(path);
}

inline void write_snapshot_bin(const std::string& filename,
                               const std::vector<double>& U,
                               const std::vector<double>& pmax_hist,
                               const std::vector<double>& Tmax_hist,
                               int Nx, int Ny,
                               double t,
                               double xL, double xR, double yL, double yR,
                               double dx, double dy) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot find snaps" + filename);

    SnapshotHeader h;
    h.Nx = Nx;
    h.Ny = Ny;
    h.t = t;
    h.xL = xL; h.xR = xR; h.yL = yL; h.yR = yR;
    h.dx = dx; h.dy = dy;
    out.write(reinterpret_cast<const char*>(&h), sizeof(SnapshotHeader));

    std::vector<double> inner(static_cast<std::size_t>(Nx) * static_cast<std::size_t>(Ny) * NV);
    std::size_t q = 0;
    for (int i = 1; i <= Nx; ++i) {
        for (int j = 1; j <= Ny; ++j) {
            for (int m = 0; m < NV; ++m) {
                inner[q++] = U[idx(i, j, m, Ny)];
            }
        }
    }
    out.write(reinterpret_cast<const char*>(inner.data()), static_cast<std::streamsize>(inner.size() * sizeof(double)));

  
    const std::size_t n_inner = static_cast<std::size_t>(Nx) * static_cast<std::size_t>(Ny);
    if (pmax_hist.size() == n_inner && Tmax_hist.size() == n_inner) {
        out.write(reinterpret_cast<const char*>(pmax_hist.data()), static_cast<std::streamsize>(pmax_hist.size() * sizeof(double)));
        out.write(reinterpret_cast<const char*>(Tmax_hist.data()), static_cast<std::streamsize>(Tmax_hist.size() * sizeof(double)));
    }
}

inline void write_run_metadata(const std::string& filename, const RunConfig& cfg, const Params& par) {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("metadata" + filename);
    out << std::setprecision(17);
    out << "gamma=" << par.gamma << "\n";
    out << "Q=" << par.Q << "\n";
    out << "k=" << par.k << "\n";
    out << "Ea=" << par.Ea << "\n";
    out << "Rgas=" << par.Rgas << "\n";
    out << "xL=" << cfg.xL << "\n";
    out << "xR=" << cfg.xR << "\n";
    out << "yL=" << cfg.yL << "\n";
    out << "yR=" << cfg.yR << "\n";
    out << "Nx=" << cfg.Nx << "\n";
    out << "Ny=" << cfg.Ny << "\n";
    out << "final_time=" << cfg.final_time << "\n";
    out << "cfl=" << cfg.cfl << "\n";
    out << "react_cfl=" << cfg.react_cfl << "\n";
    out << "save_dt=" << cfg.save_dt << "\n";
    out << "use_muscl=" << (cfg.use_muscl ? 1 : 0) << "\n";
    out << "num_threads=" << cfg.num_threads << "\n";
#ifdef _OPENMP
    out << "openmp_enabled=1\n";
#else
    out << "openmp_enabled=0\n";
#endif
}

inline void append_diagnostics_header(const std::string& filename) {
    std::ofstream out(filename);
    out << "step,t,dt,p_max,lam_max,shock_x_mean\n";
}

inline void append_diagnostics_row(const std::string& filename,
                                   int step, double t, double dt,
                                   double p_max, double lam_max, double shock_x_mean) {
    std::ofstream out(filename, std::ios::app);
    out << std::setprecision(17)
        << step << "," << t << "," << dt << ","
        << p_max << "," << lam_max << "," << shock_x_mean << "\n";
}
