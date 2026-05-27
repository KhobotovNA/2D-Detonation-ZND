#pragma once
#include <string>
#include <vector>

struct Params {
    double gamma = 1.4;
    double Q     = 20.0;
    double k     = 1.0;
    double Ea    = 20.0;
    double Rgas  = 1.0;

    double rho_a = 1.0;
    double u_a   = 0.0;
    double v_a   = 0.0;
    double p_a   = 1.0;
    double lam_a = 0.0;
};

struct ExtraMode {
    int mode = 1;
    double amp = 0.0;
    double phase = 0.0;
};

struct Hotspot {
    double center_x = 0.0;
    double center_y = 0.0;
    double sigma_x  = 1.0;
    double sigma_y  = 1.0;
    double p_amp    = 0.0;
    double v_amp    = 0.0;
    double lam_amp  = 0.0;
};

struct IgnitionConfig {
    double x_ign_left  = -50.0;
    double x_ign_right = 0.0;
    double rho_ign = 1.0;
    double u_ign   = 0.0;
    double v_ign   = 0.0;
    double p_ign   = 70.0;
    double lam_ign = 0.0;

    bool smooth = true;
    double smooth_width = 1.0;

    double edge_amp = 3.0;
    int edge_mode = 1;
    double edge_phase = 0.0;
    std::vector<ExtraMode> edge_extra_modes = {
        {2, 0.75, 0.7},
        {3, 0.50, 1.3},
        {4, 0.30, 2.2}
    };

    double p_amp = 0.08;
    double v_amp = 0.14;
    double seed_sigma_x = 4.0;
    double seed_center_x = 0.5;
    int seed_mode = 2;
    double seed_phase = 0.35;
    double seed_phase_v = 1.5707963267948966;

    std::vector<Hotspot> hotspots;
};

struct RunConfig {
    double xL = -50.0;
    double xR = 180.0;
    double yL = 0.0;
    double yR = 24.0;

    int Nx = 384;
    int Ny = 48;

    double final_time = 4.0;
    double cfl = 0.20;
    double react_cfl = 0.20;
    double save_dt = 0.50;
    double track_dt = 0.25;

    bool use_muscl = true;
    bool save_snapshots = true;


    int num_threads = 0;

    std::string output_dir = "cpp_openmp_output";
};
