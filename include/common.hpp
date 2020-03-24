#pragma once

#include <iostream>
#include <algorithm>
#include <memory>
#include <inference_engine.hpp>

/**
 * @brief This class represents a console error listener.
 *
 */
class ConsoleErrorListener : public InferenceEngine::IErrorListener {
    /**
     * @brief The plugin calls this method with a null terminated error message (in case of error)
     * @param msg Error message
     */
    void onError(const char *msg) noexcept override {
        std::clog << "Plugin message: " << msg << std::endl;
    }
};

template<typename T> struct array_deleter {
  inline void operator()(T const * p) {
    delete [] p;
  }
};

struct RGB {
  unsigned char r, g, b;
};

/* class that can hold RGB image data in a format similar to golang's image.Image */
class RGB24 {
public:
  unsigned char * pix;
  int stride;
  int x0, x1, y0, y1;

  inline int pixOffset(int x, int y) const {
    return (y - y0) * stride + (x - x0) * 3;
  }

  inline bool isInside(int x, int y) const {
    return x >= x0 && x < x1 && y >= y0 && y < y1;
  }
public:
  RGB at(int x, int y) const {
    if (!isInside(x,y)) {
      return RGB{0,0,0};
    }

    int i = pixOffset(x, y);
    return RGB{ pix[i], pix[i+1], pix[i+2] };
  }

  inline int dx() const {
    return x1 - x0;
  }
  inline int dy() const {
    return y1 - y0;
  }

  RGB24(unsigned char * pix, int stride, int x0, int y0, int x1, int y1)
    : pix(pix), stride(stride), x0(x0), y0(y0), x1(x1), y1(y1)
  { }

  // RGB(int width, int height)
  //   : pix(new unsigned char[width * height]), stride(3 * width), x0(0), y0(0), x1(width), y1(height), did_allocate(true)
  // {
  //   std::fill(pix, pix + width * height, 0);
  // }
  //
  // ~RGB() {
  //   if (did_allocate) {
  //     delete [] pix;
  //   }
  // }
};

static std::ostream &operator<<(std::ostream &os, const InferenceEngine::Version *version) {
    os << "\n\tAPI version ............ ";
    if (nullptr == version) {
        os << "UNKNOWN";
    } else {
        os << version->apiVersion.major << "." << version->apiVersion.minor;
        if (nullptr != version->buildNumber) {
            os << "\n\t" << "Build .................. " << version->buildNumber;
        }
        if (nullptr != version->description) {
            os << "\n\t" << "Description ....... " << version->description;
        }
    }
    return os;
}
