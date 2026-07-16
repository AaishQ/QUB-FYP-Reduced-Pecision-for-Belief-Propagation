//
// Created for ARM CPU - Converted from CUDA from the Benchmark BP created by Mark Van der Merwe, Fall 2018
// Author: AaishahQ

#include "serial_inference_helpers.h"
#include <utility>
#include <vector>
#include "header.h"
#include <ctime>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <fstream> 

using namespace std;

std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> infer(pgm* pgm, double epsilon, int timeout, std::vector<int> runtime_params, bool verbose) {

    int num_edges = pgm->edge_idx_to_edges_idx.size();
    int edge_size = pgm->edges.size();

    std::vector<double> workspace = pgm->edges;
    bool converged = false;

    // Random number generation here using some of the standard c++ libraries i found
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Parameters for therandom BP
    float p_low = 0.3;  // Low parallelism - updates fewer edges
    float p_high = 0.8; // High parallelism - updates more edges

    // override the default probabilities

    if (runtime_params.size() >= 2) {
        p_low = runtime_params[0] / 10.0;
        p_high = runtime_params[1] / 10.0;
    }

    std::cout << "Starting Random Belief Propagation." << std::endl;
    std::cout << "Low parallelism: " << p_low << ", High parallelism: " << p_high << std::endl;

   // convergence timing 
    auto start = std::chrono::steady_clock::now();
    std::chrono::duration<float> time_span;
    float time = 0.0;

    int iterations = 0;
    int edges_updated = 0;

    // Tracking for which edges need updates
    std::vector<double> residuals(num_edges, 10.0);
    std::vector<int> need_update(num_edges, 1);

    while (!converged && time < timeout) {
        ++iterations;

        // Determines the update probability based on convergence rate
        float p = (iterations % 10 == 0) ? p_high : p_low;

        // Updating the  edges randomly
        for (int edge_id = 0; edge_id < num_edges; ++edge_id) {
            if (residuals[edge_id] > epsilon) {
                if (dis(gen) <= p) {
                    double delta = compute_message(pgm, workspace, edge_id, false);
                    // Apply said updates
                    int edges_index = pgm->edge_idx_to_edges_idx[edge_id];
                    int size_of_message = (int)pgm->edges[edges_index];
                    for (int i = 0; i < size_of_message; ++i) {
                        pgm->edges[edges_index + 1 + i] = workspace[edges_index + 1 + i];
                    }
                    edges_updated++;
                    residuals[edge_id] = delta;
                }
            }
        }

        // Update the residuals for all the edges
        int need_update_count = 0;
        for (int edge_id = 0; edge_id < num_edges; ++edge_id) {
            
            // recalc for accurate conv

            double delta = compute_message(pgm, workspace, edge_id, false);
            residuals[edge_id] = delta;
            need_update[edge_id] = (delta > epsilon) ? 1 : 0;
            need_update_count += need_update[edge_id];
        }

        // Copy wthe workspace to the edges in prep for next iteration
        workspace = pgm->edges;

        // Check convergence - Set to true whent here are no edges that need updating
        converged = (need_update_count == 0);

        // I  print the  progress every 50 iterations
        if (verbose && iterations % 50 == 0) {
            std::cout << "Iteration " << iterations << ": " << need_update_count << " edges need update" << std::endl;
        }

        // I update the duration time here to check for any timeout conditions

        auto current_time = std::chrono::steady_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<float>>(current_time - start);
        time = time_span.count();
    }

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    auto converge_time = std::chrono::duration<double, std::milli>(diff).count();

    // Memory measurement
    long mem = 0;
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long size, resident, share, text, lib, data, dt;
        statm >> size >> resident >> share >> text >> lib >> data >> dt;
        mem = resident * 4;
        statm.close();
    }

    std::cout << "Time: " << converge_time << " ms." << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Memory: " << mem << " KB" << std::endl;
    std::cout << "Total edges updated: " << edges_updated << std::endl;

    // Computing the final marginals
    compute_marginals(pgm);

    // Multiplies all the incoming messges and then normalises to get the proper distrubition amongst nodes
    if (verbose) {
        print_doubles(pgm->marginal_rep);
    }

    // The return of all the results
    std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> results(converge_time, pgm->marginal_rep, iterations, {}, {});
    return results;
}
