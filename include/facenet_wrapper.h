#ifndef __FACENET_WRAPPER_H__
#define __FACENET_WRAPPER_H__

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct Facenet facenet;

  typedef struct classifier_request_t {
    char * data;
    unsigned long image_width;
    unsigned long image_height;
  } classifier_request;

  typedef struct classifier_response_t {
    float * embedding;
    unsigned long embedding_size;
    float duration;
  } classifier_response;

  facenet * create_classifier(char * networkFile, char * networkWeights, char * deviceName);
  int classifier_get_embedding_size(facenet * c);
  void destroy_classifier(facenet * c);
  classifier_response * classifier_do_classification(facenet * c, void * data, int stride, int x0, int y0, int x1, int y1);
  void destroy_classifier_response(classifier_response * r);

#ifdef __cplusplus
}
#endif

#endif
