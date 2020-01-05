#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

//#include <mkl.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <Eigen/QR>
#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;

template<class Scalar>
using RowMatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template<class Scalar>
using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
template<class Scalar>
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template<class Scalar>
using MappedCovMatrix = Eigen::Map<Eigen::Matrix<Scalar, 5, 5, Eigen::RowMajor>>;


template<class Scalar>
std::tuple<py::array_t<Scalar>, py::array_t<Scalar>> fast_batch_update(
        const Eigen::Ref<const RowMatrixX<Scalar>> states,
        const py::array_t<const Scalar> root_covs,
        const Eigen::Ref<const VectorX<Scalar>> measurements,
        const Eigen::Ref<const VectorX<Scalar>> loadings,
        const Scalar meas_var)
{
    py::buffer_info buffer_info = root_covs.request();
    auto *data = static_cast<Scalar *>(buffer_info.ptr);

    const auto& shape = buffer_info.shape;
    int nobs = shape[0];
    const auto& strides = buffer_info.strides;
    int outer_stride = strides[0];
    //for(const auto s : strides)
    //    std::cout << s << std::endl;

    py::array_t<Scalar> updated_states_arr({nobs, 5});
    Eigen::Map<RowMatrixX<Scalar>> updated_states((Scalar*)updated_states_arr.request().ptr, nobs, 5);

    py::array_t<Scalar> updated_root_cov_arr({nobs, 5, 5});
    auto *updated_cov_data = static_cast<Scalar*>(updated_root_cov_arr.request().ptr);

    auto meas_var_sqrt = sqrt(meas_var);

#pragma omp parallel for schedule(static)
    for(int i = 0; i < nobs; ++i){
        Eigen::Map<Eigen::Matrix<Scalar, 5, 5>> root_cov(data + outer_stride * i); //no alignement information

        Eigen::Matrix<Scalar, 6, 6> m;
        m.template block<5,5>(1,1).noalias() = root_cov.transpose();

        m(0,0) = meas_var_sqrt;
        m.row(0).template block<1,5>(0,1).noalias() = Eigen::Matrix<Scalar, 1, 5>::Zero();

        auto f_star = m.template block<5,5>(1,1) * loadings;
        m.col(0).template block<5,1>(1,0).noalias() = f_star;

        //Eigen::HouseholderQR<Eigen::Ref<Eigen::Matrix<Scalar, 6, 6>>> dec(m);
        const auto dec = m.householderQr();
        const auto& qr = dec.matrixQR();
        const Scalar sigma = qr(0,0);
        auto kalman_gain = (Scalar(1.) / sigma) * qr.template block<1,5>(0,1);

        auto residual = measurements[i] - states.row(i).dot(loadings);

        updated_states.row(i).noalias() = states.row(i) + kalman_gain * residual;
        MappedCovMatrix<Scalar> mappedCov(updated_cov_data + i * 25);
        qr.template block<5,5>(1,1).template triangularView<Eigen::Upper>().transpose().evalTo(mappedCov);
    }

    return std::make_tuple(updated_states_arr, updated_root_cov_arr);
}




PYBIND11_MODULE(kalman, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def(
            "fast_batch_update",
            fast_batch_update<double>,
            "states"_a,
            "root_covs"_a,
            "measurements"_a,
            "loadings"_a,
            "meas_var"_a,
            R"pbdoc(
        Update *state* and *root_cov with* with a *measurement*.
    Args:
        state (np.ndarray): pre-update estimate of the unobserved state vector
        root_cov (np.ndarray): lower triangular matrix square-root of the
            covariance matrix of the state vector before the update
        measurement (float): the measurement to incorporate
        loadings (np.ndarray): the factor loadings
        meas_var (float): The variance of the incorporated measurement.
    Returns:
        updated_state (np.ndarray)
        updated_root_cov (np.ndarray))pbdoc"
        );


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}


#pragma clang diagnostic pop