#pragma once

#include "common.hpp"

#include <inference_engine.hpp>
#include <ext_list.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include "half.hpp"

using std::string; 
using half_float::half;

using namespace InferenceEngine;

class Facenet {
  ConsoleErrorListener error_listener;
  InferencePlugin plugin;
  CNNNetwork network;
  ExecutableNetwork executable_network;
  InferRequest infer_request;

  string inputImageName;
  string outputName;
  int embedding_size;
  Precision precision;
  Precision outputPrecision;
public:
  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
  typedef std::chrono::duration<float> fsec;

  struct response {
    float duration;
    std::vector<float> embedding;
  };
  Facenet() {}

  Facenet(string networkFile, string networkWeights, string plugin_name, string plugin_path) {
    std::clog << "InferenceEngine: " << GetInferenceEngineVersion() << "\n";
    plugin = PluginDispatcher({ plugin_path.c_str(), "" }).getPluginByDevice(plugin_name);
    if (plugin_name == "CPU") {
      plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    } else if (plugin_name == "MYRIAD") {
      // plugin.SetConfig({{PluginConfigParams::KEY_VPU_HW_STAGES_OPTIMIZATION, "NO"}});
      // plugin.SetConfig({{"KEY_VPU_HW_STAGES_OPTIMIZATION", "NO"}});
      plugin.SetConfig({{"VPU_LOG_LEVEL", "LOG_WARNING"}});
    }
    static_cast<InferenceEngine::InferenceEnginePluginPtr>(plugin)->SetLogCallback(error_listener);

    CNNNetReader networkReader;
    networkReader.ReadNetwork(networkFile);
    networkReader.ReadWeights(networkWeights);
    CNNNetwork network = networkReader.getNetwork();

    InputsDataMap inputsInfo(network.getInputsInfo());
    if (inputsInfo.size() != 1)
      throw std::logic_error("Sample supports topologies only with 1 inputs");

    InputInfo::Ptr inputInfo = inputsInfo.begin()->second;

    auto item = inputsInfo.begin();

    if (item->second->getInputData()->getTensorDesc().getDims().size() == 4) {
      auto imageInput = item->second->getInputData();

      std::clog << "input shape: " << imageInput->getTensorDesc().getDims()[0] << ","
                                   << imageInput->getTensorDesc().getDims()[1] << ","
                                   << imageInput->getTensorDesc().getDims()[2] << ","
                                   << imageInput->getTensorDesc().getDims()[3] << "\n";

      inputImageName = item->first;
      std::clog << "input name: " << inputImageName << "\n";
      std::clog << "input layout: " << item->second->getLayout() << "\n";
      std::clog << "inpuat preprocess variant: " << item->second->getPreProcess().getMeanVariant() << "\n";
      item->second->setPrecision(Precision::U8);
      std::clog << "input precision: " << item->second->getPrecision() << "\n";
      precision = item->second->getPrecision();
      item->second->setLayout(Layout::NCHW);
      item->second->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    } else {
      throw std::logic_error("only dimensions of size 4 are allowed");
    }

    OutputsDataMap outputsInfo(network.getOutputsInfo());
    DataPtr outputInfo;
    for (const auto& out : outputsInfo) {
      // std::clog << "output type: " << out.second->creatorLayer.lock()->type << "\n";
      std::clog << "output name: " << out.first << "\n";
      std::clog << "output size: ";
      for (auto const & d : out.second->getTensorDesc().getDims()) {
        std::clog << d << ",";
      }
      std::clog << "\n";

      // if (out.second->creatorLayer.lock()->type == "Normalize") {
        outputName = out.first;
        outputInfo = out.second;
        // break;
      // }
    }

    if (outputInfo == nullptr) {
      throw std::logic_error("output info not found");
    }

    // if (plugin_name == "MYRIAD") {
    //   outputInfo->setPrecision(Precision::FP16);
    // }
    
    outputInfo->setPrecision(Precision::FP32);
    outputPrecision = outputInfo->getPrecision();
    std::clog << "output precision: " << outputInfo->getPrecision() << "\n";
    // outputInfo->setLayout(Layout::NC);

    // executable_network = plugin.LoadNetwork(network, {{"VPU_LOG_LEVEL", "LOG_DEBUG"}});
    executable_network = plugin.LoadNetwork(network, {});
    infer_request = executable_network.CreateInferRequest();

    const SizeVector outputDims = outputInfo->getTensorDesc().getDims();

    if (outputDims.size() != 2)
      throw std::logic_error("expected output dims size of 2");

    embedding_size = outputDims[1];
  }
  int get_embedding_size() const {
    return embedding_size;
  }
  response InferRGB(unsigned char * pix, int stride, int x0, int y0, int x1, int y1) {
    RGB24 rgb(pix, stride, x0, y0, x1, y1);

    return InferRGB(rgb);
  }
  response InferRGB(const RGB24 & rgb) {
    TensorDesc tdesc(precision, {1, 3, (unsigned long)rgb.dy(), (unsigned long)rgb.dx()}, InferenceEngine::Layout::NCHW);
    Blob::Ptr blob = make_shared_blob<PrecisionTrait<Precision::U8>::value_type>(tdesc);
    blob->allocate();

    PrecisionTrait<Precision::U8>::value_type* image = static_cast<PrecisionTrait<Precision::U8>::value_type*>(blob->buffer());

    for(int y = 0; y < rgb.dy(); y++) {
      for(int x = 0; x < rgb.dx(); x++) {
        // std::clog << "[ " << x << " " << y << " " << c << " ]\n";
        RGB col = rgb.at(x + rgb.x0, y + rgb.y0);

        image[0 * rgb.dy() * rgb.dx() + y * rgb.dx() + x] = col.b;
        image[1 * rgb.dy() * rgb.dx() + y * rgb.dx() + x] = col.g;
        image[2 * rgb.dy() * rgb.dx() + y * rgb.dx() + x] = col.r;
      }
    }
    infer_request.SetBlob(inputImageName, blob);

    // std::clog << "performing inference\n";
    auto t0 = Time::now();
    infer_request.Infer();
    auto t1 = Time::now();
    fsec fs = t1 - t0;
    ms d = std::chrono::duration_cast<ms>(fs);

    float duration = d.count();

    // std::clog << "getting output blob\n";
    const Blob::Ptr output_blob = infer_request.GetBlob(outputName);

    const float* embedding = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());

    // std::clog << "returning response\n";
    return response{
      duration,
      std::vector<float>(embedding, embedding + embedding_size)
    };
  }
};
