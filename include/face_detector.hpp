#pragma once

#include <iostream>
#include <inference_engine.hpp>
#include <ext_list.hpp>
#include <chrono>
#include <string>

#include "common.hpp"

using std::string;
using namespace InferenceEngine;

struct Proposal {
  float confidence;
  float label;
  float xmin;
  float ymin;
  float xmax;
  float ymax;
};

class FaceDetector {
  ConsoleErrorListener error_listener;
  InferencePlugin plugin;
  CNNNetwork network;
  ExecutableNetwork executable_network;
  InferRequest infer_request;
  Blob::Ptr imageInput;

  string imageInputName;
  string outputName;
  size_t num_channels;
  size_t image_width;
  size_t image_height;
  int maxProposalCount;
  float min_confidence;

  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
  typedef std::chrono::duration<float> fsec;
public:
  void set_min_confidence(float min_confidence) {
    this->min_confidence = min_confidence;
  }
  struct response {
    float duration; // in ms
    std::vector<Proposal> proposal;
  };
  FaceDetector() {}
  ~FaceDetector() {}
  FaceDetector(string networkFile, string networkWeights, string plugin_name, string plugin_path)
    : min_confidence(0.75)
  {
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << "\n";

    std::cout << "FaceDetector('" << networkFile << "', '" << networkWeights << "', '" << plugin_name << "')\n";
    plugin = PluginDispatcher({ plugin_path.c_str(), "" }).getPluginByDevice(plugin_name);
    if (plugin_name == "CPU") {
      plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    }
    static_cast<InferenceEngine::InferenceEnginePluginPtr>(plugin)->SetLogCallback(error_listener);

    std::cout << "reading network...\n";
    CNNNetReader networkReader;
    networkReader.ReadNetwork(networkFile);

    std::cout << "reading weights...\n";
    networkReader.ReadWeights(networkWeights);

    std::cout << "getNetwork()...\n";
    network = networkReader.getNetwork();

    std::cout << "checking inputs...\n";

    InputsDataMap inputsInfo(network.getInputsInfo());
    if (inputsInfo.size() != 1)
      throw std::logic_error("Sample supports topologies only with 1 inputs");

    InputInfo::Ptr inputInfo = inputsInfo.begin()->second;


    std::cout << "getting input info...\n";
    auto item = inputsInfo.begin();

    if (item->second->getInputData()->getTensorDesc().getDims().size() == 4) {
      imageInputName = item->first;
      item->second->setPrecision(Precision::U8);
      item->second->setLayout(Layout::NCHW);
      item->second->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    } else {
      throw std::logic_error("only dimensions of size 4 are allowed");
    }

    std::cout << "getting output info...\n";
    OutputsDataMap outputsInfo(network.getOutputsInfo());
    DataPtr outputInfo;
    for (const auto& out : outputsInfo) {
      if (out.second->creatorLayer.lock()->type == "DetectionOutput") {
        outputName = out.first;
        outputInfo = out.second;
        break;
      }
    }
    if (outputInfo == nullptr) {
      throw std::logic_error("Can't find a DetectionOutput layer in the topology");
    }
    const SizeVector outputDims = outputInfo->getTensorDesc().getDims();
    if (outputDims.size() != 4) {
      throw std::logic_error("Incorrect output dimensions for SSD model");
    }
    maxProposalCount = outputDims[2];
    if (outputDims[3] != 7) {
      throw std::logic_error("Output item should have 7 as a last dimension");
    }

    std::cout << "setting precision...\n";
    outputInfo->setPrecision(Precision::FP32);
    executable_network = plugin.LoadNetwork(network, {});
    infer_request = executable_network.CreateInferRequest();

    imageInput = infer_request.GetBlob(imageInputName);

    num_channels = imageInput->getTensorDesc().getDims()[1];
    image_width = imageInput->getTensorDesc().getDims()[3];
    image_height = imageInput->getTensorDesc().getDims()[2];

    std::clog << "[" << image_width << " " << image_height << "," << num_channels << "]\n";
  }
  size_t get_num_channels() const {
    return num_channels;
  }
  size_t get_image_width() const {
    return image_width;
  }
  size_t get_image_height() const {
    return image_height;
  }
  size_t blobSize() const {
    return num_channels * image_width * image_height;
  }
  response InferRGB(void * data, int stride, int x0, int y0, int x1, int y1) {
    return InferRGB(RGB24((unsigned char *)data, stride, x0, y0, x1, y1));
  }
  response InferRGB(const RGB24& rgb) {
    TensorDesc tdesc(Precision::U8, {1, 3, (unsigned long)rgb.dy(), (unsigned long)rgb.dx()}, InferenceEngine::Layout::NCHW);
    Blob::Ptr blob = make_shared_blob<unsigned char>(tdesc);
    blob->allocate();

    unsigned char* image = static_cast<unsigned char*>(blob->buffer());

    for(int y = 0; y < rgb.dy(); y++) {
      for(int x = 0; x < rgb.dx(); x++) {
        // std::clog << "[ " << x << " " << y << " " << c << " ]\n";
        RGB col = rgb.at(x + rgb.x0, y + rgb.y0);

        image[0 * rgb.dy() * rgb.dx() + y * rgb.dx() + x] = col.b;
        image[1 * rgb.dy() * rgb.dx() + y * rgb.dx() + x] = col.g;
        image[2 * rgb.dy() * rgb.dx() + y * rgb.dx() + x] = col.r;

        // image[c * rgb.dy() * rgb.dx() + y * rgb.dx() + x] = data[y * rgb.dx() * 3 + x * 3 + 3 - c];
      }
    }

    infer_request.SetBlob(imageInputName, blob);

    auto t0 = Time::now();
    infer_request.Infer();
    auto t1 = Time::now();
    fsec fs = t1 - t0;
    ms d = std::chrono::duration_cast<ms>(fs);

    response res;

    res.duration = d.count();

    const Blob::Ptr output_blob = infer_request.GetBlob(outputName);
    const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());

    /* Each detection has image_id that denotes processed image */
    int curProposal = 0;
    for (; curProposal < maxProposalCount; curProposal++) {
      float image_id = detection[curProposal * 7 + 0];
      if (image_id < 0) {
          break;
      }
      float label = detection[curProposal * 7 + 1];
      float confidence = detection[curProposal * 7 + 2];
      float xmin = detection[curProposal * 7 + 3];
      float ymin = detection[curProposal * 7 + 4];
      float xmax = detection[curProposal * 7 + 5];
      float ymax = detection[curProposal * 7 + 6];

      // std::clog << "det: " << confidence << " " << ymin << "\n";

      if (confidence > min_confidence) {
        res.proposal.push_back(Proposal{confidence, label, xmin, ymin, xmax, ymax});
      } else {
        break;
      }
    }
    return res;
  }
};
