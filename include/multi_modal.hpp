#pragma once

#include <type_traits>
#include <numeric>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <tuple>
#include <map>
#include <set>

using std::pair;

const double sqrt2 = 1.414213562373095;
const double pi = 3.141592653589793;
const double sqrtpi = 1.772453850905516;
const double eps = 1e-10;


template<typename X> struct norm {
  double operator()(X const & a) const {
    return std::abs(a);
  }
};
template<typename N> struct norm<std::vector<N>> {
  double operator()(std::vector<N> const & a) const {
    std::vector<N> sq = a;
    std::transform(a.begin(), a.end(), sq.begin(), [](N const & n) { return n * n; });
    return std::sqrt(std::accumulate(sq.begin(), sq.end(), 0.0));
  }
};
template<typename X, typename T> X scale(X const & x, T const & factor) {
    return x * factor;
}
template<typename U, typename T> std::vector<U> scale(std::vector<U> const & x, T const & factor) {
  std::vector<U> ret = x;
  std::transform(x.begin(), x.end(), ret.begin(), [&factor](U const & v) {
    return v * factor;
  });
  return ret;
};

namespace std {
  template<typename V> struct plus<std::vector<V>> {
    std::vector<V> operator()(std::vector<V> const & a, std::vector<V> const & b) const {
      std::vector<V> ret = a;
      std::transform(a.begin(), a.end(), b.begin(), ret.begin(), std::plus<V>());
      return ret;
    }
  };
  template<typename V> struct minus<std::vector<V>> {
    std::vector<V> operator()(std::vector<V> const & a, std::vector<V> const & b) const {
      std::vector<V> ret = a;
      std::transform(a.begin(), a.end(), b.begin(), ret.begin(), std::minus<V>());
      return ret;
    }
  };
}

template<typename X> struct distribution {
  X mean;
  double m2;
  unsigned long count;

  distribution(X const & p)
    : mean(p), m2(0.), count(1)
  { }
  distribution()
    : mean(), m2(0.), count(0)
  { }

  static distribution<X> from_standard_deviation(X const & mean, double standard_deviation, unsigned long count) {
    distribution<X> ret(mean);
    ret.m2 = standard_deviation * standard_deviation * count;
    ret.count = count;
    return ret;
  }

  double variance() const {
    return m2 / (double)count;
  }
  double standard_deviation() const {
    return std::sqrt(variance());
  }
  double likelihood(X const & x) const {
    static std::minus<X> minus;
    static norm<X> norm;

    if (count == 1) return 0.;
    return 1. - std::erf(norm(minus(mean, x)) / standard_deviation() / sqrt2);
  }

  void serialize(std::ostream & os) const {
    throw std::logic_error("not implemented");
  }
  void deserialize(std::istream & is) {
    throw std::logic_error("not implemented");
  }

  double density(double from_mean) const {
    if (m2 > 0)
      return std::exp(-from_mean * from_mean / variance() / 2.) / sqrt2 / sqrtpi / standard_deviation();
    
    if (from_mean == 0.) return 1.0;
    return 0.0;
  }
};

template<> void distribution<std::vector<double>>::serialize(std::ostream & os) const {
  if (!os) return;

  unsigned long siz = mean.size();
  os.write((const char *)&siz, sizeof(unsigned long));
  os.write((const char *)mean.data(), siz * sizeof(double));
  os.write((const char *)&m2, sizeof(double));
  os.write((const char *)&count, sizeof(unsigned long));

  // std::cout << "size: " << siz << " m2: " << m2 << " count: " << count << std::endl;
}


template<> void distribution<std::vector<float>>::serialize(std::ostream & os) const {
  if (!os) return;

  unsigned long siz = mean.size();
  os.write((const char *)&siz, sizeof(unsigned long));
  os.write((const char *)mean.data(), siz * sizeof(float));
  os.write((const char *)&m2, sizeof(double));
  os.write((const char *)&count, sizeof(unsigned long));

  // std::cout << "size: " << siz << " m2: " << m2 << " count: " << count << std::endl;
}


template<> void distribution<std::vector<double>>::deserialize(std::istream & is) {
  if (!is) return;

  unsigned long siz = 0;
  is.read((char *)&siz, sizeof(unsigned long));

  // std::cerr << "vector size: " << siz << std::endl;

  double * buf = new double[siz];
  is.read((char *)buf, siz * sizeof(double));
  is.read((char *)&m2, sizeof(double));
  is.read((char *)&count, sizeof(unsigned long));
  mean = std::vector<double>(buf, buf+siz);

  delete [] buf;
}

template<> void distribution<std::vector<float>>::deserialize(std::istream & is) {
  if (!is) return;

  unsigned long siz = 0;
  is.read((char *)&siz, sizeof(unsigned long));

  // std::cerr << "vector size: " << siz << std::endl;

  float * buf = new float[siz];
  is.read((char *)buf, siz * sizeof(float));
  is.read((char *)&m2, sizeof(double));
  is.read((char *)&count, sizeof(unsigned long));
  mean = std::vector<float>(buf, buf+siz);

  delete [] buf;
}

// template<typename T> struct distribution<std::vector<T>> {
//   std::vector<T> mean;
//   std::vector<double> m2;
//   unsigned long count;

//   distribution(std::vector<T> const & p) 
//     : mean(p), m2(p.size()), count(1) 
//   { 
//     std::fill(m2.begin(), m2.end(), 0.);
//   }

//   static distribution<std::vector<T>> from_standard_deviation(std::vector<T> const & mean, std::vector<double> const & standard_deviation, unsigned long count) {
//     std::vector<double> m2(standard_deviation);
//     std::transform(standard_deviation.begin(), standard_deviation.end(), m2.begin(), [count](auto const & stddev) -> double {
//       return stddev * stddev * (double)count;
//     });

//     return distribution<std::vector<T>>{mean, m2, count};
//   }

//   std::vector<double> variance() const {
//     std::vector<double> variance(m2);
//     std::transform(m2.begin(), m2.end(), variance.begin(), [&](auto const & m) {
//       return m / (double)count;
//     });
//     return variance;
//   }

//   double variance_determinant() const {
//     std::vector<double> var = variance();
//     return std::accumulate(var.begin(), var.end(), 1., std::multiplies<double>{});
//   }

//   double standard_deviation() const {
//     return std::sqrt(variance_determinant());
//   }

//   // double likelihood(std::vector<T> const & x) {
//   //   std::vector<double> variance(m2);
//   //   std::transform(m2.begin(), m2.end(), variance.begin(), [count](auto const & m) {
//   //     return m / (double)count;
//   //   });

//   //   std::vector<T> diff(mean);
//   //   std::transform(mean.begin(), mean.end(), x.begin(), diff.begin(), std::minus<double>{});

//   //   std::vector<double> scaled(m2);
//   //   std::transform(diff.begin(), diff.end(), variance.begin(), scaled.begin(), [count](auto const & diff, double v) {
//   //     return diff * diff / v;
//   //   });
//   //   double exp = std::accumulate(scaled.begin(), scaled.end(), 0.);

//   //   double det = std::accumulate(variance.begin(), variance.end(), 1., std::multiplies<double>{});

//   //   double g = std::exp(-exp / 2.) / 
//   // }

//   double density(std::vector<T> const & x) const {
//     auto var = variance();
//     auto det = variance_determinant();

//     if (det < eps)
//       return 0.;

//     std::vector<T> val(mean);
//     std::transform(mean.begin(), mean.end(), x.begin(), val.begin(), std::minus<double>{});
//     std::transform(val.begin(), val.end(), var.begin(), val.begin(), [](auto const & v, auto const & s) -> double {
//       return v * v / s;
//     });
//     double exponent = std::accumulate(val.begin(), val.end(), 0., std::plus<double>{});

//     return std::exp(-exponent / 2.) / std::pow(sqrtpi * sqrt2, x.size()) / std::sqrt(det);
//   }

//   std::vector<T> zero() {
//     std::vector<T> zed(mean.size());
//     std::fill(zed.begin(), zed.end(), 0.);
//     return zed;
//   }
// };

template<typename T>
std::ostream& operator<<(std::ostream & os, distribution<T> const & dist) {
  return os << "{ " << dist.mean << " / " << dist.standard_deviation() << " # " << dist.count << " }";
}

// L2 norm
template<typename X> 
double mixture_error(distribution<X> const & a, distribution<X> const & b) {
  static norm<X> norm;
  static std::minus<X> minus;

  distribution<X> c = mix(a, b);

  double mua = norm(minus(a.mean, c.mean));
  double mub = norm(minus(b.mean, c.mean));

  double alpha = (double)a.count / (double)c.count;

  double vara = a.variance();
  double varb = b.variance();
  double stda = a.standard_deviation();
  double stdb = b.standard_deviation();

  double ret = 0.;

  if (vara < eps || varb < eps) {
    return std::numeric_limits<double>::max();
  }

  ret += alpha * alpha / stda +
         (1. - alpha) * (1. - alpha) / stdb +
         1. / c.standard_deviation();
  ret /= sqrt2;

  ret +=   2. * alpha * (1. - alpha) / std::sqrt(vara + varb)
         - 2. * alpha / std::sqrt(vara + c.variance())
         - 2. * (1. - alpha) / std::sqrt(varb + c.variance());
  
  ret /= sqrt2 / sqrtpi;

  return ret;
}

// total variational distance sup | (A(+)B) - C | 
template<typename X>
double mixture_error2(distribution<X> const & a, distribution<X> const & b) {
  static norm<X> norm;
  static std::minus<X> minus;

  distribution<X> c = mix(a, b);

  double mac = norm(minus(a.mean, c.mean));
  double mbc = norm(minus(b.mean, c.mean));
  double mab = norm(minus(a.mean, b.mean));
  double alpha = (double)a.count / (double)c.count;

  double ea = std::abs(alpha * a.density(0)   + (1. - alpha) * b.density(mab) - c.density(mac));
  double eb = std::abs(alpha * a.density(mab) + (1. - alpha) * b.density(0)   - c.density(mbc));
  double ec = std::abs(alpha * a.density(mac) + (1. - alpha) * b.density(mbc) - c.density(0));

  // std::cout << "density: " << a.density(0) << "\n";

  // std::cout << "ac, bc, ab, alpha, ea, eb, ec: " << mac << " " << mbc << " " << mab << " " << alpha << " " << ea << " " << eb << " " << ec << " " << std::endl;

  return ea * a.standard_deviation() + eb * b.standard_deviation() + ec * c.standard_deviation();
}

// template<typename T>
// double mixture_error(distribution<std::vector<T>> const & a, distribution<std::vector<T>> const & b) {
//   distribution<std::vector<T>> c = mix(a, b);

//   double alpha = (double)a.count / (double)c.count;

//   double scale = std::pow(sqrtpi * sqrt2, a.mean.size());
//   double astd = std::sqrt(a.variance_determinant()),
//          bstd = std::sqrt(b.variance_determinant()),
//          cstd = std::sqrt(c.variance_determinant());

//   double ea = std::abs(alpha / scale / astd + (1. - alpha) * b.density(a.mean) - c.density(a.mean)),
//          eb = std::abs(alpha * a.density(b.mean) + (1. - alpha) / scale / bstd - c.density(b.mean)),
//          ec = std::abs(alpha * a.density(c.mean) + (1. - alpha) * b.density(c.mean) - 1. / scale / cstd);

//   return ea * astd + eb * bstd + ec * cstd;
// }


template<typename X>
distribution<X> mix(distribution<X> const & a, distribution<X> const & b) {
  static norm<X> norm;
  static std::plus<X> plus;
  static std::minus<X> minus;

  unsigned long count = a.count + b.count;

  // weight factor
  double a_left = (double)a.count / (double)count;
  double a_right = 1. - a_left;

  X mean = plus(
    scale(a.mean, a_left),
    scale(b.mean, a_right)
  );

  // distance between means
  // double left_distance  = norm(minus(mean, a.mean));
  // double right_distance = norm(minus(mean, b.mean));

  // double variance = a_left  * (a.variance() + left_distance  * left_distance) +
  //                   a_right * (b.variance() + right_distance * right_distance);

  // this should be the same as above but more efficient
  double distance = norm(minus(a.mean, b.mean));
  double variance = a_left * a.variance() 
                  + a_right * b.variance() 
                  + a_left * a_right * distance * distance;

  distribution<X> ret(mean);
  ret.m2 = variance * (double)count;
  ret.count = count;
  return ret;
}

// template<typename T>
// distribution<std::vector<T>> mix(distribution<std::vector<T>> const & a, distribution<std::vector<T>> const & b) {
//   static norm<std::vector<T>> norm;
//   static std::plus<std::vector<T>> plus;
//   static std::minus<std::vector<T>> minus;

//   unsigned long count = a.count + b.count;
//   double alpha = (double)a.count / (double)count;

//   std::vector<T> mean = plus(
//     scale(a.mean, alpha),
//     scale(b.mean, 1. - alpha)
//   );

//   std::vector<double> m2(a.m2);

//   for (int i = 0; i < m2.size(); i++) {
//     double a_dist = a.mean[i] - mean[i];
//     double b_dist = b.mean[i] - mean[i];
//     m2[i] = a.m2[i] + (double)a.count * a_dist * a_dist + 
//             b.m2[i] + (double)b.count * b_dist * b_dist;
//   }

//   distribution<std::vector<T>> ret(mean);
//   ret.m2 = m2;
//   ret.count = count;
//   return ret;
// }

template<typename X>
distribution<X> unmix(distribution<X> const & c, distribution<X> const & b) {
  static norm<X> norm;
  static std::plus<X> plus;
  static std::minus<X> minus;

  double left_factor = (double)c.count / (double)(c.count - b.count);
  double right_factor = (double)b.count / (double)(c.count - b.count);

  X mean = plus(
    scale(c.mean, -left_factor),
    scale(b.mean, right_factor)
  );

  // distance between means
  double left_distance  = norm(minus(mean, c.mean));
  double right_distance = norm(minus(mean, b.mean));

  double variance = left_factor  * (c.variance() + left_distance  * left_distance) -
                    right_factor * (b.variance() + right_distance * right_distance);

  unsigned long count = c.count - b.count;

  distribution<X> ret(mean);
  ret.m2 = variance * (double)count;
  ret.count = count;
  return ret;
}


template<typename X> class multi_modal {
  struct node {
    distribution<X> dist;
    double error;
    node * left, * right;
    unsigned long id;
  };

  node * root;
  unsigned long maximum_nodes;
  unsigned long count;
  unsigned long next_id;
private:
  void insert_helper(node * n, distribution<X> const & dist) {
    static norm<X> norm;
    static std::minus<X> minus;

    if (n->left == nullptr) {
      if (count < maximum_nodes) {
        n->left = new node(*n);
        n->left->id = next_id++;
        n->right = new node{dist, 0, nullptr, nullptr, next_id++};
        n->dist = mix(n->right->dist, n->left->dist);
        // this could be more expensive
        n->error = mixture_error(n->right->dist, n->left->dist);

        count++;
      } else {
        n->dist = mix(n->dist, dist);
      }
      return;
    }

    node ** op, ** other, ** adjust;

    auto left = norm(minus(n->left->dist.mean, dist.mean));
    auto right = norm(minus(n->right->dist.mean, dist.mean));

    if (left < right) {
      op = &(n->left);
      other = &(n->right);
    } else {
      op = &(n->right);
      other = &(n->left);
    }

    insert_helper(*op, dist);
    n->dist = mix(n->left->dist, n->right->dist);
    // this could be more expensive
    n->error = mixture_error(n->right->dist, n->left->dist);

    if (n->dist.variance() < (*op)->dist.variance()) {
      // handle the case where *op is a leaf
      if ((*op)->left == nullptr) {
        // delete this node and merge the children into a new leaf?
        // bubble it up somehow?
        // is the other side a leaf?

        if ((*other)->left != nullptr) {

        }
      } else {
        if ((*op)->left->dist.variance() < (*op)->right->dist.variance()) {
          adjust = &((*op)->left);
        } else {
          adjust = &((*op)->right);
        }

        std::swap(*other, *adjust);

        (*op)->dist = mix((*op)->right->dist, (*op)->left->dist);
        (*op)->error = mixture_error((*op)->right->dist, (*op)->left->dist);
        n->dist = mix(n->right->dist, n->left->dist);
        n->error = mixture_error(n->left->dist, n->right->dist);
      }
    }
  }

  void delete_helper(node * n) {
    std::vector<node*> stack;
    stack.push_back(n);
    while(!stack.empty()) {
      node * cur = stack.back();
      stack.pop_back();
      if (cur->left != nullptr) {
        stack.push_back(cur->left);
        cur->left = nullptr;
      }
      if (cur->right != nullptr) {
        stack.push_back(cur->right);
        cur->right = nullptr;
      }
      delete cur;
      count--;
    }
  }

  template<typename Visitor> void visit_nodes(Visitor v) const {
    std::vector<std::pair<node*,unsigned long>> stack;
    stack.push_back({root,0});
    node * cur;
    unsigned long depth;
    while(!stack.empty()) {
      std::tie(cur, depth) = stack.back();
      stack.pop_back();

      if (v(cur, depth)) {
        if (cur->left != nullptr) stack.push_back({cur->left, depth+1});
        if (cur->right != nullptr) stack.push_back({cur->right, depth+1});
      }
    }
  }

  bool extract_peaks_helper2(std::set<node*> & peaks, node * cur, node * peak_ancestor) const {
    if (cur->left == nullptr) return false;

    if (peak_ancestor == nullptr || cur->error < peak_ancestor->error) {
      peak_ancestor = cur;
    } 

    bool left = false, right = false;
    left = extract_peaks_helper2(peaks, cur->left, peak_ancestor);
    right = extract_peaks_helper2(peaks, cur->right, peak_ancestor);

    if (!left && !right && cur == peak_ancestor) {
      peaks.insert(cur);
      return true;
    }

    return left || right;
  }

  bool extract_peaks_helper(std::vector<pair<unsigned long, distribution<X>>> & peaks, node * cur, node * parent) const {
    if (parent != nullptr && cur->error < parent->error) {
      peaks.push_back({cur->id, cur->dist});
      return true;
    }

    bool left = false, right = false;
    if (cur->left != nullptr) {
      left = extract_peaks_helper(peaks, cur->left, cur);
      right = extract_peaks_helper(peaks, cur->right, cur);

      // if (left && !right) {
      //   peaks.push_back(cur->right->dist);
      // } else if (!left && right) {
      //   peaks.push_back(cur->left->dist);
      // }
    }

    // add the root if we haven't added anything else
    if (parent == nullptr && peaks.size() == 0) {
      peaks.push_back({cur->id, cur->dist});
    }

    return left || right;
  }

  bool find_peak_helper(X const & x, node *& found, node * n) const {
    static norm<X> norm;
    static std::minus<X> minus;

    if (n->left == nullptr) return false;

    //should this use probability of being
    auto left = norm(minus(n->left->dist.mean, x));
    auto right = norm(minus(n->right->dist.mean, x));

    node * chosen = n->left, * other = n->right;
    if(right < left) {
      chosen = n->right;
      other = n->left;
    }

    if (chosen->error < n->error) {
      found = chosen;
      return true;
    } 

    bool ret = find_peak_helper(x, found, chosen);

    if (!ret && other->error < n->error) {
      found = chosen;
      return true;
    }

    return ret;
  }

public:
  void deserialize(std::istream & is) {
    if (root != nullptr) {
      delete_helper(root);
      root = nullptr;
    }
    count = 0;
    next_id = 0;

    std::vector<pair<char, node*>> stack;

    is.read((char*)&count, sizeof(unsigned long));
    is.read((char*)&maximum_nodes, sizeof(unsigned long));
    is.read((char*)&next_id, sizeof(unsigned long));

    std::cout << "count: " << count << " maximum_nodes: " << maximum_nodes << " next_id: " << next_id << "\n";

    if (count == 0) return;

    root = new node();
    root->left = nullptr;
    root->right = nullptr;

    root->dist.deserialize(is);
    is.read((char*)&(root->error), sizeof(double));
    is.read((char*)&(root->id), sizeof(unsigned long));

    char dir = 'L';
    node * cur = root;
    node * n = nullptr;
    
    while (is) {
      is.read(&dir, 1);

      switch (dir) {
      case 'L':
        stack.push_back({'L', cur});
        n = new node();
        n->left = nullptr;
        n->right = nullptr;
        n->dist.deserialize(is);
        is.read((char*)&(n->error), sizeof(double));
        is.read((char*)&(n->id), sizeof(unsigned long));
        cur->left = n;
        cur = n;
        break;
      case 'R':
        stack.push_back({'R', cur});
        n = new node();
        n->left = nullptr;
        n->right = nullptr;
        n->dist.deserialize(is);
        is.read((char*)&(n->error), sizeof(double));
        is.read((char*)&(n->id), sizeof(unsigned long));
        cur->right = n;
        cur = n;
        break;
      case 'P':
        cur = stack.back().second;
        stack.pop_back();
        break;
      default:
        throw std::logic_error("direction not understood");
      }
    }
  }
  void serialize(std::ostream & os) const {
    std::vector<pair<char, node const *>> stack;

    os.write((const char *)&count, sizeof(unsigned long));
    os.write((const char *)&maximum_nodes, sizeof(unsigned long));
    os.write((const char *)&next_id, sizeof(unsigned long));

    if (root == nullptr) return;

    root->dist.serialize(os);
    os.write((const char *)&(root->error), sizeof(double));
    os.write((const char *)&(root->id), sizeof(unsigned long));
    stack.push_back({'L', root});

    char pop = 'P';

    while(!stack.empty()) {
      auto p = stack.back();
      stack.pop_back();

      if (p.first == 'L' && p.second->left != nullptr) {
        stack.push_back({'R', p.second});

        os.write(&p.first, 1);
        p.second->left->dist.serialize(os);
        os.write((const char *)&(p.second->left->error), sizeof(double));
        os.write((const char *)&(p.second->left->id), sizeof(unsigned long));

        stack.push_back({'L', p.second->left});
      } else if (p.first == 'R' && p.second->right != nullptr) {
        stack.push_back({'P', p.second});
        os.write(&p.first, 1);

        p.second->right->dist.serialize(os);
        os.write((const char *)&(p.second->right->error), sizeof(double));
        os.write((const char *)&(p.second->right->id), sizeof(unsigned long));

        stack.push_back({'L', p.second->right});
      } else {
        os.write(&pop, 1);
      }

    } 
  }

  std::vector<pair<unsigned long, distribution<X>>> extract_peaks() const {
    std::vector<pair<unsigned long, distribution<X>>> ret;
    std::set<node*> peaks;
    // extract_peaks_helper(ret, root, nullptr);
    extract_peaks_helper2(peaks, root, nullptr);

    for (node * n : peaks) {
      ret.push_back({n->id, n->dist});
    }
    return ret;
  }

  template<typename Visitor> void visit(Visitor v) const {
    visit_nodes([&v](node * n, unsigned long depth) -> bool {
      return v(n->dist, depth);
    });
  }
  template<typename Visitor> void visit_children(Visitor v) const {
    visit_nodes([&v](node * n, unsigned long depth) -> bool {
      if (n->left == nullptr) return false;

      return v(n->dist, n->left->dist, n->right->dist, depth);
    });
  }

  void insert(X const & x) {
    distribution<X> dist(x);

    if (root == nullptr) {
      root = new node{dist, 0, nullptr, nullptr};
      return;
    }
    insert_helper(root, dist);
  }

  pair<unsigned long, distribution<X>> find_peak(X const & x) {
    node * n = nullptr;
    pair<unsigned long, distribution<X>> ret{0, distribution<X>(x)};

    if (root == nullptr) {
      return ret;
    }

    if (find_peak_helper(x, n, root)) {
      return {n->id, n->dist};
    }

    return ret;
  }

  unsigned long get_count() const { return count; }

  multi_modal(unsigned long max) 
    : root(nullptr), maximum_nodes(max), count(0), next_id(0)
  {}

  multi_modal() 
    : multi_modal(std::numeric_limits<unsigned long>::max()) 
  {}

  ~multi_modal() { 
    if (root == nullptr) return;

    delete_helper(root); 
    root = nullptr;
  }
};