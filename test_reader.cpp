
#include <inference_engine.hpp>
#include "common.hpp"
#include <string>
#include <iostream>
#include <gflags/gflags.h>

using namespace InferenceEngine;

/**
 * @brief Gets filename without extension
 * @param filepath - full file name
 * @return filename without extension
 */
static std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

int main(int ac, char * av[]) {
    std::cout << GetInferenceEngineVersion() << std::endl;

    if (ac < 2) {
        std::cerr << "must pass a model file name" << std::endl;
        return -1;
    }

    std::string binFileName = fileNameNoExt(av[1]) + ".bin";

    CNNNetReader networkReader;
    networkReader.ReadNetwork(av[1]);
    networkReader.ReadWeights(binFileName);
    CNNNetwork network = networkReader.getNetwork();

    std::cout << "layer count: " << network.layerCount() << std::endl;

    return 0;
}