#include <cstdlib>
#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "params.hpp"
#include "solver.hpp"

static bool read_arg(int argc, char** argv, const std::string& key, std::string& value) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) {
            value = argv[i + 1];
            return true;
        }
    }
    return false;
}

static bool has_flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) if (argv[i] == key) return true;
    return false;
}

int main(int argc, char** argv) {
    Params par;
    RunConfig cfg;
    IgnitionConfig ign;

  
    ign.hotspots = {
        {1.5,  0.22 * (cfg.yR - cfg.yL), 2.5, 2.6,  0.20,  0.080, 0.0},
        {4.5,  0.55 * (cfg.yR - cfg.yL), 3.0, 3.0,  0.16, -0.075, 0.0},
        {7.5,  0.82 * (cfg.yR - cfg.yL), 3.5, 3.2,  0.12,  0.055, 0.0}
    };

    std::string s;
    if (read_arg(argc, argv, "--nx", s)) cfg.Nx = std::stoi(s);
    if (read_arg(argc, argv, "--ny", s)) cfg.Ny = std::stoi(s);
    if (read_arg(argc, argv, "--xL", s)) cfg.xL = std::stod(s);
    if (read_arg(argc, argv, "--xR", s)) cfg.xR = std::stod(s);
    if (read_arg(argc, argv, "--yL", s)) cfg.yL = std::stod(s);
    if (read_arg(argc, argv, "--yR", s)) cfg.yR = std::stod(s);
    if (read_arg(argc, argv, "--final", s)) cfg.final_time = std::stod(s);
    if (read_arg(argc, argv, "--cfl", s)) cfg.cfl = std::stod(s);
    if (read_arg(argc, argv, "--react-cfl", s)) cfg.react_cfl = std::stod(s);
    if (read_arg(argc, argv, "--save-dt", s)) cfg.save_dt = std::stod(s);
    if (read_arg(argc, argv, "--track-dt", s)) cfg.track_dt = std::stod(s);
    if (read_arg(argc, argv, "--out", s)) cfg.output_dir = s;
    if (read_arg(argc, argv, "--pign", s)) ign.p_ign = std::stod(s);
    if (read_arg(argc, argv, "--threads", s)) cfg.num_threads = std::stoi(s);
    if (has_flag(argc, argv, "--first-order")) cfg.use_muscl = false;
    if (has_flag(argc, argv, "--no-snapshots")) cfg.save_snapshots = false;

#ifdef _OPENMP
    if (cfg.num_threads > 0) {
        omp_set_num_threads(cfg.num_threads);
    }
    const int omp_threads = omp_get_max_threads();
    cfg.num_threads = omp_threads;
#else
    cfg.num_threads = 1;
#endif

 
    ign.hotspots[0].center_y = cfg.yL + 0.22 * (cfg.yR - cfg.yL);
    ign.hotspots[1].center_y = cfg.yL + 0.55 * (cfg.yR - cfg.yL);
    ign.hotspots[2].center_y = cfg.yL + 0.82 * (cfg.yR - cfg.yL);

    std::cout << "2D reactive Euler \n";
    std::cout << "Nx, Ny      = " << cfg.Nx << ", " << cfg.Ny << "\n";
    std::cout << "domain x    = [" << cfg.xL << ", " << cfg.xR << "]\n";
    std::cout << "domain y    = [" << cfg.yL << ", " << cfg.yR << "]\n";
    std::cout << "final_time  = " << cfg.final_time << "\n";
#ifdef _OPENMP
    std::cout << "OpenMP      = enabled, threads=" << cfg.num_threads << "\n";
#else
    std::cout << "OpenMP      = disabled at compile time\n";
#endif
    std::cout << "scheme      = HLLC + " << (cfg.use_muscl ? "MUSCL/MC" : "piecewise constant")
              << " + SSP-RK3 + Strang reaction\n";

    try {
        Solver2D solver(par, cfg, ign);
        solver.run();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
