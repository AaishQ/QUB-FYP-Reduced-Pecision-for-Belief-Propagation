//
// Further Reduced Precision implementation (8 bits) - Format: 1 sign bit, 2 exponent bits, 5 mantissa bits, Bias = 1. Range approx +/- 2^2 = +/-4
// Template based on loopy Benchmark BP
// Based on A. S. Molahosseini, J. Lee and H. Vandierendonck, "Software-Defined Number Formats for High-Speed Belief Propagation," in IEEE Transactions on Emerging Topics in Computing, vol. 13, no. 3, pp. 853-865, July-Sept. 2025
// 
// Author: AaishahQ

#include "serial_inference_helpers.h"
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>

using namespace std;

// 8-bit  structure
// I used 5 mantissa bits (paper used 6) to test the absolute limit on ARM
struct sdf8 {
    uint8_t bits;    // store in 8 bits

    sdf8() : bits(0) {}

    sdf8(double d) {
        if (d == 0.0) {
            bits = 0;
            return;
        }

        // extract sign (bit 7)
        uint8_t sign = 0;
        if (d < 0.0) {
            sign = 0x80;
            d = -d;
        }

        // Clamp to representable range - 8-bit can only handle roughly 0.015 to 4
        if (d > 3.9) d = 3.9;
        if (d < 0.015625) d = 0.015625;

        // Normalise to [1.0, 2.0) - only 2 exponent bits so range is limited
        int exp = 0;
        while (d >= 2.0 && exp < 2) {
            d = d * 0.5;
            exp++;
        }
        while (d < 1.0 && exp > -2) {
            d = d * 2.0;
            exp--;
        }

        // Bias = 1 (for 2 exponent bits), clamp to 2 bits (0-3)
        int biased_exp = exp + 1;
        if (biased_exp < 0) biased_exp = 0;
        if (biased_exp > 3) biased_exp = 3;

        // Mantissa: 5 bits - remove leading 1 and scale by 2^5 = 32
        d = d - 1.0;
        uint8_t mantissa = (uint8_t)(d * 32.0);
        if (mantissa > 31) mantissa = 31;

        // pack: sign (bit7), exponent (bits6-5), mantissa (bits4-0)
        bits = sign | (biased_exp << 5) | mantissa;
    }

    // convert back from 8-bit to double
    double to_double() const {
        if (bits == 0) return 0.0;

        int sign = (bits & 0x80) ? -1 : 1;
        int biased_exp = (bits >> 5) & 0x3;
        uint8_t mantissa = bits & 0x1F;

        double d = 1.0 + (mantissa / 32.0);
        int exp = biased_exp - 1;

        if (exp > 0) {
            for (int i = 0; i < exp; i++) d = d * 2.0;
        }
        else if (exp < 0) {
            for (int i = 0; i < -exp; i++) d = d * 0.5;
        }

        return sign * d;
    }

    // overload operators 
    sdf8 operator+(const sdf8& other) const {
        return sdf8(this->to_double() + other.to_double());
    }

    sdf8 operator*(const sdf8& other) const {
        return sdf8(this->to_double() * other.to_double());
    }

    sdf8& operator=(double d) {
        *this = sdf8(d);
        return *this;
    }
};

// message computation using 8-bit type
float compute_msg_8bit(pgm* pgm, vector<sdf8>& workspace, int edge_id, bool write_to_edges) {
    int edges_index = pgm->edge_idx_to_edges_idx[edge_id];
    int size_of_message = (int)pgm->edges[edges_index];

    int start = pgm->pgm_graph->edge_idx_to_incoming_edges[edge_id];
    int end = pgm->pgm_graph->edge_idx_to_incoming_edges[edge_id + 1];

    for (int i = 0; i < size_of_message; i++) {
        workspace[edges_index + 1 + i] = sdf8(0.0);
    }

    int edge_factor_start = pgm->edge_idx_to_edge_factors_idx[edge_id];
    int node_factor_start = pgm->edge_idx_to_node_factors_idx[edge_id];

    int first_arg = pgm->edge_factors[edge_factor_start] == edge_id;
    int source_size;
    if (first_arg) {
        source_size = pgm->edge_factors[edge_factor_start + 1];
    }
    else {
        source_size = pgm->edge_factors[edge_factor_start + 2];
    }
    edge_factor_start += 3;

    for (int setting = 0; setting < size_of_message; setting++) {
        sdf8 value(0.0);

        for (int s = 0; s < source_size; s++) {
            sdf8 partial(1.0);

            int factor_idx;
            if (first_arg) {
                factor_idx = edge_factor_start + (s * size_of_message) + setting;
            }
            else {
                factor_idx = edge_factor_start + s + (setting * source_size);
            }
            partial = partial * sdf8(pgm->edge_factors[factor_idx]);
            partial = partial * sdf8(pgm->node_factors[node_factor_start + 1 + s]);

            for (int msg_idx = start; msg_idx < end; msg_idx++) {
                int msg_start = pgm->edge_idx_to_edges_idx[pgm->pgm_graph->edge_incoming_edges[msg_idx]];
                partial = partial * workspace[msg_start + 1 + s];
            }

            value = value + partial;
        }

        workspace[edges_index + 1 + setting] = value;
    }

    sdf8 sum(0.0);
    for (int i = 0; i < size_of_message; i++) {
        sum = sum + workspace[edges_index + 1 + i];
    }

    double sum_double = sum.to_double();
    if (sum_double > 1e-10) {
        for (int i = 0; i < size_of_message; i++) {
            double val = workspace[edges_index + 1 + i].to_double();
            workspace[edges_index + 1 + i] = sdf8(val / sum_double);
        }
    }

    double diff_sum = 0.0;
    for (int i = 0; i < size_of_message; i++) {
        double diff = workspace[edges_index + 1 + i].to_double() - pgm->edges[edges_index + 1 + i];
        diff_sum = diff_sum + diff * diff;
    }
    float delta = (float)sqrt(diff_sum);

    if (write_to_edges) {
        for (int i = 0; i < size_of_message; i++) {
            pgm->edges[edges_index + 1 + i] = workspace[edges_index + 1 + i].to_double();
        }
    }

    return delta;
}

tuple<float, vector<double>, int, vector<pair<int, int>>, vector<pair<float, int>>>
infer(pgm* pgm, double epsilon, int timeout, vector<int> runtime_params, bool verbose) {

    std::cout << "FURTHER REDUCED PRECISION FOR LOOPY BELIEF PROPAGATION" << std::endl;
    std::cout << "Format: 1 sign, 2 exponent, 5 mantissa" << std::endl;
    std::cout << "NOTE: May not work on all graphs :(" << std::endl;

    // save original edges so I can restore them after inference
    vector<double> original_edges = pgm->edges;

    // create workspace using 8-bit type
    vector<sdf8> workspace(pgm->edges.size());
    for (size_t i = 0; i < pgm->edges.size(); i++) {
        workspace[i] = sdf8(pgm->edges[i]);
    }

    int num_edges = pgm->edge_idx_to_edges_idx.size();
    bool converged = false;
    float eps = (float)epsilon;

    auto start = chrono::steady_clock::now();
    int iterations = 0;

    while (!converged) {
        iterations++;
        converged = true;

        for (int e = 0; e < num_edges; e++) {
            float delta = compute_msg_8bit(pgm, workspace, e, true);
            if (delta > eps) {
                converged = false;
            }
        }

        auto now = chrono::steady_clock::now();
        float elapsed = chrono::duration_cast<chrono::seconds>(now - start).count();
        if (elapsed > timeout) break;
    }

    auto end = chrono::steady_clock::now();
    float time_ms = chrono::duration<double, milli>(end - start).count();

    // Memory measurement - HAD TO DO THIS BEFORE restoring original_edges
    // measure peak memory while the 8-bit workspace is still alive
    long mem = 0;
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long size, resident, share, text, lib, data, dt;
        statm >> size >> resident >> share >> text >> lib >> data >> dt;
        mem = resident * 4;
        statm.close();
    }

    // restore original edges before computing marginals
    pgm->edges = original_edges;

    compute_marginals(pgm);

    std::cout << "Time: " << time_ms << " ms" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Memory: " << mem << " KB" << std::endl;

    return make_tuple(time_ms, pgm->marginal_rep, iterations,
        vector<pair<int, int>>(), vector<pair<float, int>>());
}
