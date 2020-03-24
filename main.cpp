#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <random>
#include "write_jpeg.hpp"
#include "face_detector.hpp"
#include "facenet.hpp"
#include "multimodal.hpp"

using std::min;
using std::max;

template<typename T>
void write_embedding(std::string filename, std::vector<T> const & embedding) {
  std::ofstream fs;
  fs.open(filename, std::ios::out | std::ios::trunc);
  
  fs << "[";
  for (auto i = embedding.begin(); i != embedding.end(); i++) {
    if (i != embedding.begin()) {
      fs << ", ";
    }
    fs << *i;
  }
  fs << "]";

  fs.close();
}


int main(int ac, char *av[]) {
/*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d1{-500, 0.001};
  std::normal_distribution<> d2{300, 0.001};
  std::normal_distribution<> d3{1000, 0.001};

  multi_modal<float> mm;

  for(int n = 0; n < 3000; n++) {
    switch (gen() % 3) {
    case 0: mm.insert(d1(gen)); break;
    case 1: mm.insert(d2(gen)); break;
    case 2: mm.insert(d3(gen)); break;
    }
  }

  auto important = mm.best_mixture();

  for(auto const & dist : important) {
    std::cout << "[" << dist.mean << "; " << dist.standard_deviation() << "]\n";
  }

  if (important.size() != 3)
    mm.print(std::cout, 0.001 * 0.001);
  // mm.print_categories(std::cout);
  return 0;
*/
  std::string plugin_name = "CPU";
  if (ac > 1) {
    plugin_name = av[1];
    std::clog << plugin_name;
  } else {
    plugin_name = "CPU";
  }

  std::istream * in = &std::cin;
  bool need_io_cleanup = false;
  if (ac > 2) {
    std::ifstream * fs = new std::ifstream();
    fs->open(av[2], std::ios::in);
    in = fs;
    need_io_cleanup = true;
  }

  Facenet * facenet = nullptr;

  FaceDetector detector;

  if (plugin_name == "MYRIAD") {
    detector = FaceDetector(
      "../face-detection-model/FP16/face-detection-adas-0001.xml",
      "../face-detection-model/FP16/face-detection-adas-0001.bin",
      "MYRIAD",
      "/opt/intel/openvino/deployment_tools/inference_engine/lib/");

    facenet = new Facenet(
      "../resnet50_128_caffe/FP16/resnet50_128.xml",
      "../resnet50_128_caffe/FP16/resnet50_128.bin",
      "MYRIAD",
      "/opt/intel/openvino/deployment_tools/inference_engine/lib/");

  } else {
    detector = FaceDetector(
      "../face-detection-model/FP32/face-detection-adas-0001.xml",
      "../face-detection-model/FP32/face-detection-adas-0001.bin",
      "CPU",
      "/opt/intel/openvino/deployment_tools/inference_engine/lib/");

    facenet = new Facenet(
      "../resnet50_128_caffe/FP32/resnet50_128.xml",
      "../resnet50_128_caffe/FP32/resnet50_128.bin",
      "CPU",
      "/opt/intel/openvino/deployment_tools/inference_engine/lib/");

  }
  detector.set_min_confidence(0.75);


  // auto image_width = detector.get_image_width();
  // auto image_height = detector.get_image_height();
  auto image_width = 1920;
  auto image_height = 1080;
  auto num_channels = detector.get_num_channels();
  char * read_data = new char[image_width * image_height * num_channels];

  int max_faces = 1024;
  int id = 0;

  while (in->read(read_data, image_width * image_height * num_channels)) {
    auto res = detector.InferRGB(read_data, 3 * image_width, 0, 0, image_width, image_height);

    std::clog << "duration: " << res.duration << "\n";

    for (auto & p : res.proposal) {
      std::clog << "prob = " << p.confidence <<
        "    (" << p.xmin << "," << p.ymin << ")-(" << p.xmax << "," << p.ymax << ")" << std::endl;

      int x0 = p.xmin * image_width;
      int x1 = p.xmax * image_width;
      int y0 = p.ymin * image_height;
      int y1 = p.ymax * image_height;

      if (x0 < 0 || x1 >= image_width || y0 < 0 || y1 >= image_height || x0 >= x1 || y0 >= y1) {
        continue;
      }

      std::cout << x0 << "," << y0 << "-" << x1 << "," << y1 << std::endl;

      // extract the pixels of the face
      int width = x1 - x0;
      int height = y1 - y0;

      unsigned char * extracted = new unsigned char[num_channels * width * height];
      
      for (int c = 0; c < num_channels; c++) {
        for(int w = 0, w1 = x0; w1 < x1; w++, w1++) {
          for(int h = 0, h1 = y0; h1 < y1; h++, h1++) {
            extracted[h * width * num_channels + w * num_channels + c] =
              read_data[h1 * image_width * num_channels + w1 * num_channels + c];
          }
        }
      }

      
      std::stringstream filename;
      std::stringstream embeddingName;

      filename << "output/test" << std::setfill('0') << std::setw(5) << id << ".jpg";
      embeddingName << "output/test" << std::setfill('0') << std::setw(5) << id << ".json";
      id = (id + 1) % max_faces;
      
      write_jpeg(extracted, height, width, filename.str().c_str(), 90);

      auto res = facenet->InferRGB(extracted, width * 3, 0, 0, width, height);

      write_embedding(embeddingName.str(), res.embedding);
      
      
      delete [] extracted;
    }
    std::clog << "\n";

  }

  delete [] read_data;
  delete facenet;

  if (need_io_cleanup) {
    delete in;
  }

  return 0;
}
