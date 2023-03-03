#include <caratheodory.hpp>

#include <numeric>
#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>

namespace exd {

/**
 * @brief Algorithm 1: Caratheodory(P, u)
 * find S and w s.t. sum(u[i] * p[i]) == sum(w[i] * s[i])
 */
void caratheodory(const Eigen::MatrixXd& P_, const Eigen::VectorXd& u_, Eigen::MatrixXd& S_, Eigen::VectorXd& w_, Eigen::VectorXi& u_in_w, int N) {
  if (N <= 0) {
    N = P_.rows() + 1;
  }

  if (P_.cols() <= N) {
    S_ = P_;
    w_ = u_;
    u_in_w = Eigen::VectorXi::Ones(P_.cols());
    return;
  }

  Eigen::MatrixXd P = P_;
  Eigen::VectorXd u = u_;
  u_in_w = Eigen::VectorXi::Ones(P_.cols());

  int last_PN = -1;
  while (P.cols() > N) {
    if (last_PN > 0 && last_PN == P.cols()) {
      std::cerr << "warning: PN=" << last_PN << " didn't decrease (caratheodory, N=" << N << ")" << std::endl;
      break;
    }
    last_PN = P.cols();

    Eigen::MatrixXd A = P.rightCols(P.cols() - 1).colwise() - P.col(0);

    // find v s.t. Av = 0
    // Use LU decomposition to find the nullspace of A
    Eigen::VectorXd vz = A.fullPivLu().kernel().col(0);

    // Use SVD to find v (much slower than LU)
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    // Eigen::VectorXd vz = svd.matrixV().col(P.cols() - 2);

    Eigen::VectorXd v(P.cols());
    v.bottomRows(P.cols() - 1) = vz;
    v[0] = -v.bottomRows(P.cols() - 1).sum();

    double alpha = std::numeric_limits<double>::max();
    for (int i = 0; i < P.cols(); i++) {
      if (v[i] > 0) {
        alpha = std::min(alpha, u[i] / v[i]);
      }
    }

    Eigen::VectorXd w_all = u - alpha * v;

    int pos_count = (w_all.array() > 0.0).count();
    if (pos_count == w_all.size()) {
      // std::cerr << "warning: all weights have non-zero values (maybe a numerical error)!!" << std::endl;
      // std::cerr << "       : drop the smallest weight (" << w_all.minCoeff() << ")" << std::endl;
      *std::min_element(w_all.begin(), w_all.end()) = 0.0;
      pos_count--;
    }

    Eigen::MatrixXd S(P.rows(), pos_count);
    Eigen::VectorXd w(pos_count);

    pos_count = 0;
    for (int i = 0; i < w_all.size(); i++) {
      if (w_all[i] > 0.0) {
        S.col(pos_count) = P.col(i);
        w[pos_count] = w_all[i];
        pos_count++;
      }
    }

    int cursor = 0;
    for (int i = 0; i < u_in_w.size(); i++) {
      if (u_in_w[i]) {
        if (w_all[cursor] <= 0.0) {
          u_in_w[i] = 0;
        }
        cursor++;
      }
    }

    P = std::move(S);
    u = std::move(w);
  }

  S_ = std::move(P);
  w_ = std::move(u);
}

/**
 * @brief Algorithm 2: Fast-Caratheodory(P, u, k)
 * find C and w s.t. sum(u[i] * p[i]) == sum(w[i] * c[i])
 * note P and u are disruptively used as computation buffers
 */
void fast_caratheodory(Eigen::MatrixXd& P, Eigen::VectorXd& u, int k_, Eigen::MatrixXd& C_, Eigen::VectorXd& w_, Eigen::VectorXi& u_indices_, int N) {
  if (N <= 0) {
    N = P.rows() + 1;
  }

  if (P.cols() <= N) {
    C_ = P;
    w_ = u;
    return;
  }

  u_indices_.resize(P.cols());
  std::iota(u_indices_.data(), u_indices_.data() + u_indices_.size(), 0);

  int trial = 0;
  int last_PN = -1;
  while (P.cols() > N) {
    if (trial++ > 20) {
      std::cerr << "warning: Too many trials!! Something wrong!! trial=" << trial << std::endl;
    }

    if (last_PN > 0 && last_PN == P.cols()) {
      std::cerr << "PN=" << last_PN << " didn't decrease!! (fast_caratheodory, N=" << N << ")" << std::endl;
      break;
    }
    last_PN = P.cols();

    int k = std::min<int>(k_, P.cols());

    Eigen::MatrixXd P_sub = Eigen::MatrixXd::Zero(P.rows(), k);
    Eigen::VectorXd u_sub = Eigen::VectorXd::Zero(k);

    std::vector<int> indices;
    indices.reserve(k);
    indices.emplace_back(0);
    for (int i = 0; i < k; i++) {
      indices.emplace_back(indices[i] + (P.cols() + i) / k);
    }

    for (int i = 0; i < k; i++) {
      size_t begin = indices[i];
      size_t end = indices[i + 1];

      for (int j = begin; j < end; j++) {
        u_sub[i] += u[j];
        P_sub.col(i) += u[j] * P.col(j);
      }
      P_sub.col(i) *= 1.0 / u_sub[i];
    }

    int n_per_cluster = indices[indices.size() - 1] - indices[indices.size() - 2];
    int N_sub = P.rows() + 1;
    if (N_sub * n_per_cluster < N) {
      N_sub = N / n_per_cluster;
    }

    Eigen::MatrixXd S_sub;
    Eigen::VectorXd w_sub;
    Eigen::VectorXi u_in_w_;
    caratheodory(P_sub, u_sub, S_sub, w_sub, u_in_w_, N_sub);

    int k_cursor = 0;
    int num_points = 0;
    for (int i = 0; i < k; i++) {
      if (u_in_w_[i] == 0) {
        continue;
      }

      size_t begin = indices[i];
      size_t end = indices[i + 1];

      for (int j = begin; j < end; j++) {
        num_points++;
      }

      k_cursor++;
    }

    Eigen::VectorXi u_indices(num_points);
    Eigen::MatrixXd C(P.rows(), num_points);
    Eigen::VectorXd w(num_points);

    k_cursor = 0;
    num_points = 0;
    for (int i = 0; i < k; i++) {
      if (u_in_w_[i] == 0) {
        continue;
      }

      size_t begin = indices[i];
      size_t end = indices[i + 1];

      double sum_weights = 0.0;
      for (int j = begin; j < end; j++) {
        sum_weights += u[j];
      }

      for (int j = begin; j < end; j++) {
        u_indices[num_points] = u_indices_[j];
        C.col(num_points) = P.col(j);
        w[num_points] = w_sub[k_cursor] * u[j] / sum_weights;
        num_points++;
      }

      k_cursor++;
    }

    P = std::move(C);
    u = std::move(w);
    u_indices_ = std::move(u_indices);
  }

  C_ = std::move(P);
  w_ = std::move(u);
}

/**
 * @brief Algorithm 2: Fast-Caratheodory(P, u, k)
 * find S s.t. A*A^T == S*S^T
 */
void fast_caratheodory_matrix_(const Eigen::MatrixXd& A, int k, Eigen::VectorXi& indices, Eigen::VectorXd& w, Eigen::MatrixXd& S, int N) {
  Eigen::MatrixXd P(A.rows() * A.rows(), A.cols());
  for (int i = 0; i < A.cols(); i++) {
    Eigen::MatrixXd p = A.col(i) * A.col(i).transpose();

    P.col(i) = Eigen::Map<Eigen::VectorXd>(p.data(), A.rows() * A.rows());
  }

  Eigen::VectorXd u = Eigen::VectorXd::Constant(P.cols(), 1.0 / P.cols());

  Eigen::MatrixXd C;
  Eigen::VectorXi u_indices;
  fast_caratheodory(P, u, k, C, w, u_indices, N);

  S.resize(A.rows(), w.size());
  for (int i = 0; i < w.size(); i++) {
    S.col(i) = std::sqrt(A.cols() * w[i]) * A.col(u_indices[i]);
  }
  indices = u_indices;

  for (int i = 0; i < w.size(); i++) {
    w[i] = std::sqrt(A.cols() * w[i]);
  }
}

void fast_caratheodory_matrix(const Eigen::MatrixXd& A, int k, Eigen::VectorXi& indices, Eigen::VectorXd& w, Eigen::MatrixXd& S, int N) {
  Eigen::MatrixXd S_;
  fast_caratheodory_matrix_(A.transpose(), k, indices, w, S_, N);
  S = S_.transpose();
}

// Non-compact representation version
void fast_caratheodory_quadric_(const Eigen::MatrixXd& A, const Eigen::VectorXd& e, int k, Eigen::VectorXi& indices, Eigen::VectorXd& w, int target_N) {
  const int D = A.rows();
  const int N = A.cols();

  Eigen::MatrixXd P(D * D + D + 1, N);

  for (int i = 0; i < N; i++) {
    Eigen::Map<Eigen::MatrixXd>(P.col(i).data(), D, D) = A.col(i) * A.col(i).transpose();
    P.col(i).middleRows(D * D, D) = A.col(i) * e[i];
    P.col(i)[D * D + D] = e[i] * e[i];
  }

  Eigen::VectorXd u = Eigen::VectorXd::Constant(N, 1.0 / N);

  Eigen::MatrixXd C;
  fast_caratheodory(P, u, k, C, w, indices, target_N);

  w *= N;
}

void fast_caratheodory_quadratic(const Eigen::Matrix<double, -1, 6>& A, const Eigen::VectorXd& e, int k, Eigen::VectorXi& indices, Eigen::VectorXd& w, int target_N) {
  const int D = 6;
  const int N = A.rows();

  const int NH = 21;  // Number of upper triangular elements of H
  const int M = NH + D + 1;

  Eigen::MatrixXd P = Eigen::MatrixXd::Constant(M, N, std::nan(""));

  for (int i = 0; i < N; i++) {
    Eigen::Matrix<double, 6, 6> H = A.row(i).transpose() * A.row(i);

    int k = 0;
    for (int r = 0; r < D; r++) {
      for (int c = r; c < D; c++) {
        P(k++, i) = H(r, c);
      }
    }

    P.col(i).middleRows(NH, D) = A.row(i) * e[i];
    P.col(i)[NH + D] = e[i] * e[i];
  }

  Eigen::VectorXd u = Eigen::VectorXd::Constant(N, 1.0 / N);

  Eigen::MatrixXd C;
  fast_caratheodory(P, u, k, C, w, indices, target_N);

  w *= N;
}

}  // namespace exd