#pragma once

#include <Eigen/Core>

namespace exd {

/**
 * @brief Fast-Caratheodory-Matrix (Algorithm 2 in "Fast and Accurate Least-Mean-Squares Solvers")
 *        This function finds a weighted subset of P such that P^T*P = S^T*S
 * @param P       [in]  Input matrix R^{N x D}
 * @param k       [in]  Number of clusters for Fast-Caratheodory
 * @param indices [out] Indices of selected data in P
 * @param w       [out] Weights for the selected data
 * @param S       [out] Weighted subset of P
 * @param N       [in]  Target output size
*/
void fast_caratheodory_matrix(const Eigen::MatrixXd& P, int k, Eigen::VectorXi& indices, Eigen::VectorXd& w, Eigen::MatrixXd& S, int N);

/**
 * @brief Fast-Caratheodory-Quadratic (Algorithm 3 in [Koide, IROS2023]).
 *        This function finds finds a weighted subset of input residuals that exactly recovers the original quadratic error function.
 * @param J        [in]  Jacobian matrix R^{N x 6}
 * @param e        [in]  Residual vector R^N
 * @param k        [in]  Number of clusters for Fast-Caratheodory (e.g., 64)
 * @param indices  [out] Indices of selected residuals
 * @param w        [out] Weights for the selected residuals
 * @param target_N [in]  Target output size (must be >= 29)
*/
void fast_caratheodory_quadratic(const Eigen::Matrix<double, -1, 6>& J, const Eigen::VectorXd& e, int k, Eigen::VectorXi& indices, Eigen::VectorXd& w, int target_N);

}  // namespace exd
