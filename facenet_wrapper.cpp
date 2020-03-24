#include "facenet_wrapper.h"

#include "facenet.hpp"

#include <string>
#include <algorithm>
#include <iostream>

using std::string;

facenet * create_classifier(char * networkFile, char * networkWeights, char * deviceName) {
  return new Facenet(string(networkFile), string(networkWeights), string(deviceName), string());
}
void destroy_classifier(facenet * c) {
  delete c;
}
int classifier_get_embedding_size(facenet * c) {
  if (c == nullptr) return 0;

  return c->get_embedding_size();
}
classifier_response * classifier_do_classification(facenet * net, void * data, int stride, int x0, int y0, int x1, int y1) {
  // std::clog << "do_classification\n";

  // std::clog << "InferRGB\n";
  auto res = net->InferRGB((unsigned char *)data, stride, x0, y0, x1, y1);

  // std::clog << "constructing response\n";
  classifier_response * ret = new classifier_response{
    new float[res.embedding.size()],
    res.embedding.size(),
    res.duration
  };

  // std::clog << "copying data out\n";
  std::copy(res.embedding.begin(), res.embedding.end(), ret->embedding);

  return ret;
}
void destroy_classifier_response(classifier_response * r) {
  delete r->embedding;
  r->embedding = nullptr;
  delete r;
}
