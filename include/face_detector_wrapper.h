#ifndef __FACE_DETECTOR_H__
#define __FACE_DETECTOR_H__


#ifdef __cplusplus
extern "C" {
#endif 

  typedef struct FaceDetector detector;

  struct detection_tag {
    float confidence;
    float label;
    float xmin, xmax, ymin, ymax;
  };

  typedef struct detection_tag detection;

  struct response_tag {
    unsigned long num_detections;
    detection * detections;
  };

  typedef struct response_tag response;

  detector * detector_create(
    const char * networkFile,
    const char * networkWeights,
    const char * deviceName);

  void detector_set_min_confidence(detector * d, float min_confidence);
  response * detector_do_inference(detector * d, void * pix, int stride, int x0, int y0, int x1, int y1);
  void detector_destroy_response(response * res);
  void detector_destroy(detector * d);

#ifdef __cplusplus
}
#endif

#endif
