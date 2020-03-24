#pragma once

#include <type_traits>
#include <numeric>
#include <cmath>
#include <iostream>
#include <limits>

const double sqrt2 = 1.414213562373095;
const float max_variance = std::numeric_limits<float>::max();
const double min_liklihood = 1e-4;

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
  X ret = x;
  std::transform(x.begin(), x.end(), ret.begin(), [&factor](typename X::value_type const & v) {
    return v * factor;
  });
  return ret;
};
template<typename T> float scale(float const & x, T const & factor) {
  return factor * x;
}

namespace std {
  template<typename V> struct plus<std::vector<V>> {
    std::vector<V> operator()(std::vector<V> const & a, std::vector<V> const & b) const {
      std::vector<V> ret = a;
      std::transform(a.begin(), a.end(), b.begin(), ret.begin(), std::plus<V>());
      return ret;
    }
  };
}

template<typename X> struct distribution {
  X mean;
  double m2;
  size_t count;

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
    return std::erf(norm(minus(mean, x)) / standard_deviation() / sqrt2);
  }
};

template<typename X>
distribution<X> mix(distribution<X> const & a, distribution<X> const & b) {
  static norm<X> norm;
  static std::plus<X> plus;
  static std::minus<X> minus;

  size_t count = a.count + b.count;

  // weight factor
  double a_left = (double)a.count / (double)count;
  double a_right = 1. - a_left;

  X mean = plus(
    scale(a.mean, a_left),
    scale(b.mean, a_right)
  );

  // distance between means
  double left_distance  = norm(minus(mean, a.mean));
  double right_distance = norm(minus(mean, b.mean));

  double variance = a_left  * (a.variance() + left_distance  * left_distance) +
                    a_right * (b.variance() + right_distance * right_distance);

  return distribution<X>{mean, variance * (double)count, count};
}

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

  size_t count = c.count - b.count;

  return distribution<X>{mean, variance * (double)count, count};
}

// template<typename X>
// class mixture_model {
// private:
//   struct node {
//     distribution<X> dist;
//     node * left, * right;
//   };
//
//   node * root;
//
//   void insert_helper(distribution<X> const & dist, node * n) {
//     // assume n is already adjusted
//     if (n->left == nullptr) {
//       n->right = *n;
//       n->left = new node{dist, nullptr, nullptr};
//
//       n->dist = mix(n->dist, dist);
//       return;
//     }
//
//     auto new_dist = mix(n->dist, dist);
//     auto left_mixture = mix(dist, n->left->dist);
//     auto right_mixture = mix(dist, n->right->dist);
//
//
//
//     // which mixture is better?
//
//   }
// public:
//   void insert(X const & x) {
//     auto dist = distribution<X>{x, 0., 1};
//     if (root == nullptr) {
//       root = new node{, nullptr, nullptr};
//       return;
//     }
//
//     insert_helper(dist, root);
//   }
//
//   ~mixture_model() {
//     if (root == nullptr) return;
//
//     std::vector<node*> stack;
//     stack.push_back(root);
//     while(!stack.empty()) {
//       node * cur = stack.back();
//       stack.pop_back();
//
//       if (cur->left) stack.push_back(cur->left);
//       if (cur->right) stack.push_back(cur->right);
//
//       delete cur;
//     }
//   }
// };



template<typename T>
struct euclidean_distance {
  typedef typename T::value_type value_type;

  value_type operator()(T const & a, T const & b) const {
    T sq = std::inner_product(
      a.begin(), a.end(),
      b.begin(),
      0,
      [](value_type const & x, value_type const & y) -> value_type {
        return (x - y) * (x - y);
      },
      std::plus<typename T::value_type>()
    );
    return std::sqrt(sq);
  }
};

template<> struct euclidean_distance<float> {
  typedef float value_type;

  value_type operator()(value_type const & a, value_type const & b) const {
    return std::abs(a - b);
  }
};
template<> struct euclidean_distance<double> {
  typedef double value_type;

  value_type operator()(value_type const & a, value_type const & b) const {
    return std::abs(a - b);
  }
};

template<typename T> struct subtract_vector {
  T operator()(T const & a, T const & b) const {
    T ret = a;
    std::transform(a.begin(), a.end(), b.begin(), ret.begin(), std::minus<typename T::value_type>());
    return ret;
  }
};

template<typename T> struct euclidian_norm {
  typedef typename T::value_type value_type;

  value_type operator()(T const & a) const {
    value_type sq = std::accumulate(a.begin(), a.end(), 0., [](value_type const & init, value_type const & x) {
      return init + x * x;
    });
    return std::sqrt(sq);
  }
};


// DIST is a functor (x,x) -> R and satifies the triangle inequality
template<typename X, typename DIST = euclidean_distance<X>>
class multi_modal {
  typedef typename std::result_of<DIST(X,X)>::type distance_type;

  struct node {
    size_t count;
    X mean;
    distance_type m2; // used for online variance calculation
    node * left, * right;

    distance_type variance() const {
      return m2 / (distance_type)count;
    }
    distance_type standard_deviation() const {
      return std::sqrt(variance());
    }
    distance_type likelihood_inside(X const & x) const {
      static DIST dist;

      if (count == 1) return 0.;
      return std::erf(dist(mean, x) / standard_deviation() / sqrt2);
    }
    void include_vector(X const & x) {
      static std::plus<X> plus;
      static DIST dist;

      auto delta1 = dist(mean, x);
      mean = scale(plus(scale(mean, (double)count), x), 1./((double)count + 1));
      // std::cout << "mean: " << mean << "\n";
      auto delta2 = dist(mean, x);

      m2 += delta1 * delta2;
      count++;
    }
    distance_type importance() const {
      double importance = 0.0;
      if (variance() > 0) {
        importance = (double)count / standard_deviation();
        // importance = (double)count / variance();
      }
      return importance;
    }
    // this means that the children have changed, and we need to recalculate the values
    void recalculate() {
      static std::plus<X> plus;
      static DIST dist;

      // std::clog << "node:  [" << mean << " - " << count << "]\n";
      distance_type var;
      multi_modal<X,DIST>::combined(left, right, mean, count, var);

      m2 = var * (distance_type)count;

      // std::clog << "node:  E[X] = " << mean << " VAR[X] = " << variance() << " # " << count << "]\n";
      // std::clog << "   l:  E[X] = " << left->mean << " VAR[X] = " << left->variance() << " # " << left->count << "]\n";
      // std::clog << "   r:  E[X] = " << right->mean << " VAR[X] = " << right->variance() << " # " << right->count << "]\n";
      // std::clog << "parent: " << variance() << " children: " << left->variance() << ", " << right->variance() << "\n";

      if (variance() < left->variance() || variance() < right->variance()) {
        // this addition only makes the estimate better--
        // throw std::logic_error("variance relationship was not preserved");
      }
    }
  };

  static void combined(node * left, node * right, X & mean, size_t & count, distance_type & variance) {
    static std::plus<X> plus;
    static DIST dist;

    count = left->count + right->count;
    mean = scale(
      plus(
        scale(left->mean, (double)left->count),
        scale(right->mean, (double)right->count)
      ), 1./((double)count)
    );

    // weight factor
    distance_type a_left = (distance_type)left->count / (distance_type)count;
    distance_type a_right = 1. - a_left;

    // distance between means
    distance_type left_distance  = dist(mean, left->mean);
    distance_type right_distance = dist(mean, right->mean);

    variance = a_left  * (left->variance()  + left_distance  * left_distance) +
               a_right * (right->variance() + right_distance * right_distance);

    // std::clog << "[ " << mean << " ; " << count << " ; " << variance << "]\n";
  }

  node * root;
  size_t maximum_nodes;
  size_t count;

  void insert_helper(node * insert, node ** pn) {
    // if this is a leaf, move the parent to right, insert at left, then recalculate node.
    node * n = *pn;

    if (n->left == nullptr) {
      if (count < maximum_nodes) {
        n->right = new node(*n);
        n->left = insert;

        n->recalculate();
        count++;
      } else {
        size_t count;
        X mean;
        distance_type variance;

        combined(insert, n, mean, count, variance);
        n->count = count;
        n->mean = mean;
        n->m2 = variance * (double)count;
        // clean up
        delete insert;
      }
      return;
    }

    distance_type left_likelihood = 0., right_likelihood = 0.;
    left_likelihood = n->left->likelihood_inside(insert->mean);
    right_likelihood = n->right->likelihood_inside(insert->mean);

    // add it to the one to make it the most concentrated
    node * adjusted, ** other;
    if (left_likelihood <= right_likelihood) {
      insert_helper(insert, &n->left);
      adjusted = n->left;
      other = &n->right;
    } else {
      insert_helper(insert, &n->right);
      adjusted = n->right;
      other = &n->left;
    }
    n->recalculate();

    // did we maintain our goal of variance always decreasing as we go down the tree?
    if (n->variance() < adjusted->variance()) {
      // we need to figure out what to do here-- let's see what happens if we do nothing...
      if (adjusted->left == nullptr) return;

      node * temp = *other;
      if (adjusted->left->variance() < adjusted->right->variance()) {
        *other = adjusted->right;
        adjusted->right = temp;
      } else {
        *other = adjusted->left;
        adjusted->left = temp;
      }

      adjusted->recalculate();
      n->recalculate();
    }
  }

  void print_helper(std::ostream & os, float min_variance, node * n, int depth) const {
    if (n->variance() < min_variance) return;

    for(int d = depth; d > 0; d--) os << " ";

    os << "n[" << n->mean << ";" << n->variance() << " " << n->count << " -- " << n->importance() << "]\n";
    if (n->left != nullptr) print_helper(os, min_variance, n->left, depth+1);
    if (n->right != nullptr) print_helper(os, min_variance, n->right, depth+1);
  }

  template<typename NodeVisitor> void visit_nodes(NodeVisitor visitor) const {
    if (root == nullptr) return;

    std::vector<node*> stack;
    stack.push_back(root);

    while(!stack.empty()) {
      node * cur = stack.back();
      stack.pop_back();

      if (!visitor(cur)) continue;

      if (cur->left != nullptr) stack.push_back(cur->left);
      if (cur->right != nullptr) stack.push_back(cur->right);
    }
  }

  void delete_helper(node * n) {
    if (n == nullptr) return;

    std::vector<node*> stack;
    stack.push_back(n);
    while (!stack.empty()) {
      node * cur = stack.back();
      stack.pop_back();

      if (cur->right != nullptr) {
        stack.push_back(cur->right);
      }
      if (cur->left != nullptr) {
        stack.push_back(cur->left);
      }

      delete cur;
    }
  }
public:
  void reorganize() {
    auto mixture = best_mixture();

  }

  std::vector<distribution<X>> best_mixture() const {
    std::vector<distribution<X>> ret;
    if (root == nullptr) {
      return ret;
    }

    std::vector<std::pair<node *, double>> stack;
    stack.push_back({root, root->importance()});

    node * cur = nullptr;
    distance_type parent_importance;

    // drill down until importance starts decreasing
    while(!stack.empty()) {
      std::tie(cur, parent_importance) = stack.back();
      stack.pop_back();

      if (cur->left == nullptr) {
        ret.push_back({cur->mean, cur->m2, cur->count});
        continue;
      }

      bool important = true;
      if (cur->left->importance() > parent_importance) {
        stack.push_back({cur->left, cur->left->importance()});
        important = false;
      }
      if (cur->right->importance() > parent_importance) {
        stack.push_back({cur->right, cur->right->importance()});
        important = false;
      }

      if (important) {
        ret.push_back({cur->mean, cur->m2, cur->count});
      }
    }

    return ret;
  }

  void print_categories(std::ostream & os) const {
    visit_nodes([&os](node * n) {
      if (n->left == nullptr) return false;

      auto left_likelihood = n->left->likelihood_inside(n->mean);
      auto right_likelihood = n->right->likelihood_inside(n->mean);

      if (false) {
        os << "n [" << left_likelihood << "-" << right_likelihood << " ; " << n->mean << " ; " << n->standard_deviation() << " ; " << n->count << "]\n";
      }
      return true;
    });
  }

  // visitor should take X const & value, double standard_deviation, int count
  // can return false to stop descent
  template<typename Visitor>
  void visit_all(Visitor visitor) const {
    visit_nodes([&visitor](node * cur) {
      return visitor(cur->mean, cur->standard_deviation(), cur->count);
    });
  }

  void print(std::ostream & os, float min_variance) const {
    if (root != nullptr) print_helper(os, min_variance, root, 0);
  }
  void insert(X const & x) {
    // std::clog << "inserting: " << x << "\n";
    node * n = new node{1, x, 0., nullptr, nullptr};
    if (root == nullptr) {
      root = n;
    } else {
      insert_helper(n, &root);
    }
  }

  multi_modal() : root(nullptr), maximum_nodes(1000), count(0) {}
  multi_modal(multi_modal<X,DIST> const & rhs)
    : count(rhs.count), maximum_nodes(rhs.maximum_nodes), root(nullptr)
  {
    *this = rhs;
  }

  multi_modal<X,DIST> & operator=(multi_modal<X,DIST> const & rhs) {
    if (&rhs == this) return *this;

    delete_helper(root);
    count = rhs.count;
    maximum_nodes = rhs.maximum_nodes;
    root = nullptr;

    if (rhs.root == nullptr) return *this;

    std::vector<std::pair<node **,node *>> stack;
    stack.push_back({&root, rhs.root});
    node * right, ** cur;
    while(!stack.empty()) {
      std::tie(cur, right) = stack.back();
      stack.pop_back();

      *cur = new node{right->count, right->mean, right->m2, nullptr, nullptr};

      if (right->left != nullptr)  stack.push_back({&(*cur)->left, right->left});
      if (right->right != nullptr) stack.push_back({&(*cur)->right, right->right});
    }

    return *this;
  }

  ~multi_modal() {
    delete_helper(root);
  }
};
