# caratheodory2

This is a supplementary matrial of our IROS2023 paper to reproduce the numerical validation experiment.

## Build

```bash
git clone https://github.com/koide3/caratheodory2 -b devel
mkdir caratheodory2/build && cd caratheodory2/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Run

```bash
cd build
./caratheodory_test
```

## Result

```bash
N=30000
k=64
M=256
|g|=206  time=7.62872[msec]
|g|=209  time=7.69499[msec]
|g|=206  time=7.63177[msec]
...

|g|=208  time=7.60994[msec]
|g|=207  time=7.65339[msec]
|g|=206  time=7.61943[msec]
max_error=3.27418e-11
```

## Implemented functions

```cpp
/**
 * @brief Fast-Caratheodory-Matrix (Algorithm 2 in [Maalouf, NeurIPS2019])
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
```

More details will be made available once the paper gets accepted.