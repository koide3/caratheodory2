#include <chrono>
#include <iostream>
#include <caratheodory.hpp>

// Extract row vectors specified by indices
Eigen::MatrixXd extract_subset(const Eigen::MatrixXd& A, const Eigen::VectorXi& indices) {
  const int D = A.cols();
  Eigen::MatrixXd S(indices.size(), D);
  for (int i = 0; i < indices.size(); i++) {
    S.row(i) = A.row(indices[i]);
  }
  return S;
}

// Entry point
int main(int argc, char** argv) {
  const int N = 30000;  // Number of input residuals
  const int k = 64;     // Number of clusters for Fast-Caratheodory
  const int M = 256;    // Target output size (must be >= 29)
  // const int M = 29;  // Minimum coreset

  std::cout << "N=" << N << std::endl;
  std::cout << "k=" << k << std::endl;
  std::cout << "M=" << M << std::endl;

  double max_error = 0.0;
  for (int i = 0; i < 100; i++) {
    // Generate random input
    const Eigen::Matrix<double, -1, 6> J = Eigen::Matrix<double, -1, 6>::Random(N, 6);
    const Eigen::VectorXd e = Eigen::VectorXd::Random(N);

    // Run Fast-Caratheodory-Quadratic
    Eigen::VectorXi indices;
    Eigen::VectorXd w;
    const auto t1 = std::chrono::high_resolution_clock::now();
    exd::fast_caratheodory_quadratic(J, e, k, indices, w, M);
    const auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "|g|=" << w.size() << "  time=" << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6 << "[msec]" << std::endl;

    // Validation
    // Original quadratic function parameters
    const Eigen::MatrixXd H = J.transpose() * J;
    const Eigen::VectorXd b = J.transpose() * e;
    const double c = e.transpose() * e;

    // Quadratic function parameters of the extracted subset
    const Eigen::MatrixXd S = extract_subset(J, indices);
    const Eigen::VectorXd eS = extract_subset(e, indices);
    const Eigen::MatrixXd H_ = S.transpose() * w.asDiagonal() * S;
    const Eigen::MatrixXd b_ = S.transpose() * w.asDiagonal() * eS;
    const double c_ = eS.transpose() * w.asDiagonal() * eS;

    // Compute appxoimation errors
    const double error_H = (H - H_).array().abs().maxCoeff();
    const double error_b = (b - b_).array().abs().maxCoeff();
    const double error_c = std::abs(c - c_);
    const double error = std::max(error_H, std::max(error_b, error_c));
    max_error = std::max(max_error, error);

    if (error > 1e-10) {
      // Never reach here
      std::cerr << "--- H ---" << std::endl << H << std::endl;
      std::cerr << "--- H_ ---" << std::endl << H_ << std::endl;

      std::cerr << "--- b ---" << std::endl << b.transpose() << std::endl;
      std::cerr << "--- b_ ---" << std::endl << b_.transpose() << std::endl;

      std::cerr << "--- c ---" << std::endl << c << std::endl;
      std::cerr << "--- c_ ---" << std::endl << c_ << std::endl;
      abort();
    }
  }

  std::cout << "max_error=" << max_error << std::endl;

  return 0;
}