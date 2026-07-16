//
// Created by Mark Van der Merwe, Summer 2018
//

#include "header.h"
#include "Tests/test_helpers.h"
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <fstream>  // ADDED for memory reading
#include <sstream>

// Run inference on the provided graph file. Graph representation can be found summarized in Representation/pgm.cpp
// Input: txt graph file, timeout, and runtime parameters.
// Output: marginal distributions of requested nodes.
int main(int argc, char* argv[]) {

    // Parse inputs.
    if (argc < 4) { // Author AaishahQ: I've changed the argument to 4 minimum
        std::cout << "Usage:\n   ./main.out <graph filename> <timeout> [...runtime params]\n   Threshold values to test: 0.5, 0.1, 0.01, 0.001, 0.0001\n Graph filename should be a txt file containing the graph parameters. Runtime parameters should be integer to parameterize the run." << std::endl;
        return -1;
    }
    std::string filename = argv[1];
    int timeout = std::stoi(argv[2]);
    double threshold = std::stod(argv[3]); // Author 40339022: Now this allows to get the threshhold from CMD
    std::vector<int> runtime_params;
    for (int argc_iter = 4; argc_iter < argc; ++argc_iter) { // Author 40339022: Changed to index 4
        runtime_params.push_back(std::stoi(argv[argc_iter]));
    }

    // Author AaishahQ: Returning the testing output
    std::cout << "Convergence Threshold Test" << std::endl;
    std::cout << "Graph: " << filename << std::endl;
    std::cout << "Timeout: " << timeout << "s" << std::endl;
    std::cout << "Threshold (ε): " << threshold << std::endl; // This is the value to be changed
    if (!runtime_params.empty()) {
        std::cout << "Runtime params: ";
        for (int p : runtime_params) std::cout << p << " ";
        std::cout << std::endl;
    }

    // Read in the graph from the file.
    pgm* test_pgm = new pgm(filename);

    // Debug - output graph representation.
    // test_pgm->print();

     // Author AaishahQ: Memory measurement before inference
    long mem_before = 0;
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long size, resident, share, text, lib, data, dt;
        statm >> size >> resident >> share >> text >> lib >> data >> dt;
        mem_before = resident * 4;
        statm.close();
    }

    // Run inference on graph.
    std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> results = infer(test_pgm, threshold, timeout, runtime_params, false);  // CHANGED: verbose = false

    // Author AaishahQ:Memory measurement after inference
    long mem_after = 0;
    std::ifstream statm2("/proc/self/statm");
    if (statm2.is_open()) {
        long size, resident, share, text, lib, data, dt;
        statm2 >> size >> resident >> share >> text >> lib >> data >> dt;
        mem_after = resident * 4;
        statm2.close();
    }

    float runtime = std::get<0>(results);
    std::vector<double> marginals = std::get<1>(results);
    int iterations = std::get<2>(results);

    // Calculate memory used (peak is mem_after during inference)
    long mem_used = mem_after;

    std::cout << "Runtime: " << runtime << " ms." << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Memory: " << mem_used << " KB" << std::endl;
    std::cout << "Total values: " << marginals.size() << std::endl;

    // Show only first few values for verification
    if (marginals.size() > 0) {
        std::cout << "First 30 values:" << std::endl;
        for (int i = 0; i < std::min(30, (int)marginals.size()); ++i) {
            std::cout << marginals[i] << ", ";
        }
        if (marginals.size() > 30) {
            std::cout << "... (and " << marginals.size() - 30 << " more)";
        }
        std::cout << std::endl;
    }

    free(test_pgm);

    return 0;
}
