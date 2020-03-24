#include "face_detector_wrapper.h"
#include "face_detector.hpp"

#include <string>
#include <algorithm>

using std::string;
using std::transform;


detector * detector_create(
  const char * networkFile,
  const char * networkWeights,
  const char * deviceName)
{
  return new FaceDetector(string(networkFile), string(networkWeights), string(deviceName), "");
}

response * detector_do_inference(detector * f, void * pix, int stride, int x0, int y0, int x1, int y1) {
  auto req = f->InferRGB(pix, stride, x0, y0, x1, y1);

  detection * dets = new detection[req.proposal.size()];
  transform(req.proposal.begin(), req.proposal.end(), dets, [](Proposal & prop) -> detection {
    return detection{
      prop.confidence,
      prop.label,
      prop.xmin, prop.xmax,
      prop.ymin, prop.ymax
    };
  });

  return new response{req.proposal.size(), dets};
}
void detector_destroy_response(response * res) {
  // std::clog << "destroying response\n";
  delete [] res->detections;
  delete res;
}
void detector_destroy(detector * f) {
  delete f;
}
