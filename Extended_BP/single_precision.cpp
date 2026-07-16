//
// Single Precision implementation (32 bits) - Based on the Loopy BP algorithm where double has been replaced with float
// Author: AaishahQ


#include "serial_inference_helpers.h"
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>  

using namespace std;

// Utilises compute_message from serial_inference_helpers.cpp here but with floats instead
// Computes a single message from one edge to another, float precision for all  calculations
float compute_message_float(pgm* pgm, vector<float>& workspace, int edge_id, bool write_to_edges) {

    int edges_index = pgm->edge_idx_to_edges_idx[edge_id];
    int size_of_message = (int)pgm->edges[edges_index];

    int start = pgm->pgm_graph->edge_idx_to_incoming_edges[edge_id];
    int end = pgm->pgm_graph->edge_idx_to_incoming_edges[edge_id + 1];

    // clear workspace for this message
    for (int i = 0; i < size_of_message; i++) {
        workspace[edges_index + 1 + i] = 0.0f;
    }

    int edge_factor_start = pgm->edge_idx_to_edge_factors_idx[edge_id];
    int node_factor_start = pgm->edge_idx_to_node_factors_idx[edge_id];

    // Check if this edge is the first or second argument of the factor function
    int first_arg = pgm->edge_factors[edge_factor_start] == edge_id;
    int source_size;
    if (first_arg) {
        source_size = pgm->edge_factors[edge_factor_start + 1];
    }
    else {
        source_size = pgm->edge_factors[edge_factor_start + 2];
    }
    edge_factor_start += 3;

    //  computation loop main message
    for (int setting = 0; setting < size_of_message; setting++) {
        float value = 0.0f;

        for (int s = 0; s < source_size; s++) {
            float partial = 1.0f;

            int factor_idx;
            if (first_arg) {
                factor_idx = edge_factor_start + (s * size_of_message) + setting;
            }
            else {
                factor_idx = edge_factor_start + s + (setting * source_size);
            }
            partial = partial * (float)pgm->edge_factors[factor_idx];

            partial = partial * (float)pgm->node_factors[node_factor_start + 1 + s];

            // Multiply by all incoming messages
            for (int msg_idx = start; msg_idx < end; msg_idx++) {
                int msg_start = pgm->edge_idx_to_edges_idx[pgm->pgm_graph->edge_incoming_edges[msg_idx]];
                partial = partial * workspace[msg_start + 1 + s];
            }

            value = value + partial;
        }

        workspace[edges_index + 1 + setting] = value;
    }

    // Here we normalise the message by making it sum to 1
    float sum = 0.0f;
    for (int i = 0; i < size_of_message; i++) {
        sum = sum + workspace[edges_index + 1 + i];
    }

    // Avoid division by zero
    if (sum > 0.0f) {
        for (int i = 0; i < size_of_message; i++) {
            workspace[edges_index + 1 + i] = workspace[edges_index + 1 + i] / sum;
        }
    }

    // calculate how much the message changed (delta / residual)
    float diff_sum = 0.0f;
    for (int i = 0; i < size_of_message; i++) {
        float diff = workspace[edges_index + 1 + i] - (float)pgm->edges[edges_index + 1 + i];
        diff_sum = diff_sum + diff * diff;
    }
    float delta = sqrt(diff_sum);

    // If it is requested, a copy  of workspace back to main edges array
    if (write_to_edges) {
        for (int i = 0; i < size_of_message; i++) {
            pgm->edges[edges_index + 1 + i] = workspace[edges_index + 1 + i];
        }
    }

    return delta;
}

// Main inference function -> same signature as double version
tuple<float, vector<double>, int, vector<pair<int, int>>, vector<pair<float, int>>>
infer(pgm* pgm, double epsilon, int timeout, vector<int> runtime_params, bool verbose) {

    std::cout << "SINGLE PRECISION FOR LOOPY BELIEF PROPAGATION" << std::endl;

    // Convert all edges from double to float for workspace
    // this is where the main precision reduction happens
    vector<float> workspace(pgm->edges.begin(), pgm->edges.end());

    int num_edges = pgm->edge_idx_to_edges_idx.size();
    bool converged = false;
    float eps = (float)epsilon;

    auto start = chrono::steady_clock::now();
    int iterations = 0;

    while (!converged) {
        iterations++;
        converged = true;

        // Updates every edge
        for (int e = 0; e < num_edges; e++) {
            float delta = compute_message_float(pgm, workspace, e, true);
            if (delta > eps) {
                converged = false;
            }
        }

        auto now = chrono::steady_clock::now();
        float elapsed = chrono::duration_cast<chrono::seconds>(now - start).count();
        if (elapsed > timeout) {
            cout << "Timeout reached after " << elapsed << " seconds" << endl;
            break;
        }
    }

    auto end = chrono::steady_clock::now();
    float time_ms = chrono::duration<double, milli>(end - start).count();

    // Copy the final messages back to pgm for marginal computation
    // convert float back to double as we copy
    for (size_t i = 0; i < workspace.size(); i++) {
        pgm->edges[i] = workspace[i];
    }

    // Compute final marginals by using  pgm->edges which just updated
    compute_marginals(pgm);

    // measure memory usage after inference
    long mem = 0;
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long size, resident, share, text, lib, data, dt;
        statm >> size >> resident >> share >> text >> lib >> data >> dt;
        mem = resident * 4;
        statm.close();
    }

    std::cout << "Time: " << time_ms << " ms" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Memory: " << mem << " KB" << std::endl;

    // Return the results;  time, marginals, iterations, empty convergence vectors
    return make_tuple(time_ms, pgm->marginal_rep, iterations,
        vector<pair<int, int>>(), vector<pair<float, int>>());
}
