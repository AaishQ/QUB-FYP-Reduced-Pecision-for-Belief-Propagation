//
// Reduced Precision implementation (16 bits) - Format: 1 sign bit, 3 exponent bits, 12 mantissa bits, Bias = 3. Range approx +/- 2^4 = +/-16
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

// 16-bit custom float structure
// I chose 12 mantissa bits to keep 3.6 decimal digits of precision, enough for BP
struct sdf16 {
    uint16_t bits;   // store everything in 16 bits

    sdf16() : bits(0) {}

    // convert from double to my custom 16-bit format
    sdf16(double d) {
        if (d == 0.0) {
            bits = 0;
            return;
        }

        // extract sign bit (bit 15)
        uint16_t sign = 0;
        if (d < 0.0) {
            sign = 0x8000;
            d = -d;
        }

        // Clamp to representable range - prevents overflow during conversion
        if (d > 16.0) d = 16.0;
        if (d < 0.0001) d = 0.0001;

        // Normalise to [1.0, 2.0) by adjusting exponent
        int exp = 0;
        while (d >= 2.0 && exp < 4) {
            d = d * 0.5;
            exp++;
        }
        while (d < 1.0 && exp > -4) {
            d = d * 2.0;
            exp--;
        }

        // Bias = 3 (for 3 exponent bits), clamp to 3 bits (0-7)
        int biased_exp = exp + 3;
        if (biased_exp < 0) biased_exp = 0;
        if (biased_exp > 7) biased_exp = 7;

        // Mantissa: 12 bits - remove the leading 1 and scale by 2^12
        d = d - 1.0;
        uint16_t mantissa = (uint16_t)(d * 4096.0);
        if (mantissa > 4095) mantissa = 4095;

        // pack sign, exponent, mantissa into 16 bits
        bits = sign | (biased_exp << 12) | mantissa;
    }

    // convert back from my 16-bit format to double
    double to_double() const {
        if (bits == 0) return 0.0;

        // unpack
        int sign = (bits & 0x8000) ? -1 : 1;
        int biased_exp = (bits >> 12) & 0x7;
        uint16_t mantissa = bits & 0xFFF;

        // reconstruct: 1.mantissa * 2^(biased_exp - 3)
        double d = 1.0 + (mantissa / 4096.0);
        int exp = biased_exp - 3;

        if (exp > 0) {
            for (int i = 0; i < exp; i++) d = d * 2.0;
        }
        else if (exp < 0) {
            for (int i = 0; i < -exp; i++) d = d * 0.5;
        }

        return sign * d;
    }

    // overload + and * so I can use sdf16 in the existing BP code
    sdf16 operator+(const sdf16& other) const {
        return sdf16(this->to_double() + other.to_double());
    }

    sdf16 operator*(const sdf16& other) const {
        return sdf16(this->to_double() * other.to_double());
    }

    sdf16& operator=(double d) {
        *this = sdf16(d);
        return *this;
    }
};

// same compute_message logic as original but with sdf16 instead of double
float compute_message_sdf16(pgm* pgm, vector<sdf16>& workspace, int edge_id, bool write_to_edges) {
    int edges_index = pgm->edge_idx_to_edges_idx[edge_id];
    int size_of_message = (int)pgm->edges[edges_index];

    int start = pgm->pgm_graph->edge_idx_to_incoming_edges[edge_id];
    int end = pgm->pgm_graph->edge_idx_to_incoming_edges[edge_id + 1];

    for (int i = 0; i < size_of_message; i++) {
        workspace[edges_index + 1 + i] = sdf16(0.0);
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
        sdf16 value(0.0);

        for (int s = 0; s < source_size; s++) {
            sdf16 partial(1.0);

            int factor_idx;
            if (first_arg) {
                factor_idx = edge_factor_start + (s * size_of_message) + setting;
            }
            else {
                factor_idx = edge_factor_start + s + (setting * source_size);
            }
            partial = partial * sdf16(pgm->edge_factors[factor_idx]);
            partial = partial * sdf16(pgm->node_factors[node_factor_start + 1 + s]);

            for (int msg_idx = start; msg_idx < end; msg_idx++) {
                int msg_start = pgm->edge_idx_to_edges_idx[pgm->pgm_graph->edge_incoming_edges[msg_idx]];
                partial = partial * workspace[msg_start + 1 + s];
            }

            value = value + partial;
        }

        workspace[edges_index + 1 + setting] = value;
    }

    sdf16 sum(0.0);
    for (int i = 0; i < size_of_message; i++) {
        sum = sum + workspace[edges_index + 1 + i];
    }

    double sum_double = sum.to_double();
    if (sum_double > 1e-10) {
        for (int i = 0; i < size_of_message; i++) {
            double val = workspace[edges_index + 1 + i].to_double();
            workspace[edges_index + 1 + i] = sdf16(val / sum_double);
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

    std::cout << "REDUCED PRECISION FOR LOOPY BELIEF PROPAGATION" << std::endl;
    std::cout << "Format: 1 sign, 3 exponent, 12 mantissa (based on Molahosseini et al.)" << std::endl;

    // Store original edges to restore later (marginals need double)
    // i need this because the sdf16 workspace will overwrite pgm->edges during inference
    vector<double> original_edges = pgm->edges;

    // create workspace using my custom 16-bit type
    vector<sdf16> workspace(pgm->edges.size());
    for (size_t i = 0; i < pgm->edges.size(); i++) {
        workspace[i] = sdf16(pgm->edges[i]);
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
            float delta = compute_message_sdf16(pgm, workspace, e, true);
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

    // Memory measurement - DO THIS BEFORE restoring original_edges
    // i measure here so i capture the peak memory of the smaller sdf16 workspace
    long mem = 0;
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long size, resident, share, text, lib, data, dt;
        statm >> size >> resident >> share >> text >> lib >> data >> dt;
        mem = resident * 4;
        statm.close();
    }

    // Restore original edges before computing marginals
    // The SDF16 version corrupted pgm->edges
    pgm->edges = original_edges;

    compute_marginals(pgm);

    std::cout << "Time: " << time_ms << " ms" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Memory: " << mem << " KB" << std::endl;

    return make_tuple(time_ms, pgm->marginal_rep, iterations,
        vector<pair<int, int>>(), vector<pair<float, int>>());
}
