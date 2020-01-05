import pdb

import numpy as np
import pandas as pd
from numba import jit


def pandas_update(state, root_cov, measurement, loadings, meas_var):
    """Update *state* and *root_cov* with with a *measurement*.

    Args:
        state (pd.Series): pre-update estimate of the unobserved state vector
        root_cov (pd.DataFrame): lower triangular matrix square-root of the
            covariance matrix of the state vector before the update
        measurement (float): the measurement to incorporate
        loadings (pd.Series): the factor loadings
        meas_var(float): variance of the measurement error
    Returns:
        updated_state (pd.Series)
        updated_root_cov (pd.DataFrame)

    """
    # pdb.set_trace()
    expected_measurement = state.dot(loadings)
    residual = measurement - expected_measurement
    f_star = root_cov.T.dot(loadings)
    first_row = pd.DataFrame(
        data=[np.sqrt(meas_var)] + [0] * len(loadings), index=[0] + list(state.index)
    ).T
    other_rows = pd.concat([f_star, root_cov.T], axis=1)
    m = pd.concat([first_row, other_rows])
    r = np.linalg.qr(m, mode="r")
    root_sigma = r[0, 0]
    kalman_gain = pd.Series(r[0, 1:], index=state.index) / root_sigma
    updated_root_cov = pd.DataFrame(
        data=r[1:, 1:], columns=state.index, index=state.index,
    ).T
    updated_state = state + kalman_gain * residual

    return updated_state, updated_root_cov


def pandas_batch_update(states, root_covs, measurements, loadings, meas_var):
    """Call pandas_update repeatedly.

    Args:
        states (pd.DataFrame)
        root_covs (list)
        measurements (pd.Series)
        loadings (pd.Series)
        meas_var (float)
    Returns:
        updated_states (pd.DataFrame)
        updated_root_covs (list)

    """
    out_states = []
    out_root_covs = []
    for i in range(len(states)):
        updated_state, updated_root_cov = pandas_update(
            state=states.loc[i],
            root_cov=root_covs[i],
            measurement=measurements[i],
            loadings=loadings,
            meas_var=meas_var,
        )
        out_states.append(updated_state)
        out_root_covs.append(updated_root_cov)
    out_states = pd.concat(out_states, axis=1).T
    return out_states, out_root_covs


# class LoopBody:
#     def __init__(self, states, root_covs, measurements, loadings, meas_var):
#         self.states = states
#         self.root_covs = root_covs
#         self.measurements =  measurements
#         self.loadings = loadings
#         self.meas_var = meas_var
#
#     #@jit(nopython = True)
#     def __call__(self, i):
#          return numba_update(
#             state=self.states[i,:],
#             root_cov=self.root_covs[i,:],
#             measurement=self.measurements[i],
#             loadings=self.loadings,
#             meas_var=self.meas_var,
#         )


@jit(nopython=True)
def fast_batch_update(states, root_covs, measurements, loadings, meas_var):
    """Update state estimates for a whole dataset.

    Let nstates be the number of states and nobs the number of observations.

    Args:
        states (np.ndarray): 2d array of size (nobs, nstates)
        root_covs (np.ndarray): 3d array of size (nobs, nstates, nstates)
        measurements (np.ndarray): 1d array of size (nobs)
        loadings (np.ndarray): 1d array of size (nstates)
        meas_var (float):

    Returns:
        updated_states (np.ndarray): 2d array of size (nobs, nstates)
        updated_root_covs (np.ndarray): 3d array of size (nobs, nstates, nstates)
     """

    out_states = 0 * states
    out_root_covs = 0 * root_covs

    # r = Parallel(n_jobs=-1)(delayed(loop)(i) for i in range(len(states)))

    # body = LoopBody(states, root_covs, measurements, loadings, meas_var)

    # pool = multiprocessing.Pool()
    # outs = pool.map(body, range(len(states)))

    # Parallel(n_jobs=8)(delayed(body)(i) for i in range(len(states)))

    for i in range(len(states)):
        updated_state, updated_root_cov = numba_update(
            state=states[i, :],
            root_cov=root_covs[i, :],
            measurement=measurements[i],
            loadings=loadings,
            meas_var=meas_var,
        )
        out_states[i, :] = updated_state
        out_root_covs[i, :] = updated_root_cov

    return out_states, out_root_covs


def numpy_update(state, root_cov, measurement, loadings, meas_var):
    """Update *state* and *root_cov with* with a *measurement*.
    Args:
        state (np.ndarray): pre-update estimate of the unobserved state vector
        root_cov (np.ndarray): lower triangular matrix square-root of the
            covariance matrix of the state vector before the update
        measurement (float): the measurement to incorporate
        loadings (np.ndarray): the factor loadings
        meas_var (float): The variance of the incorporated measurement.
    Returns:
        updated_state (np.ndarray)
        updated_root_cov (np.ndarray)
    """
    expected_measurement = np.dot(state, loadings)
    residual = measurement - expected_measurement
    f_star = np.transpose(np.dot(np.transpose(root_cov), loadings)).reshape(-1, 1)
    first_row = np.array([np.sqrt(meas_var)] + [0] * len(loadings))
    first_row = first_row.reshape((1, -1))

    other_rows = np.concatenate((f_star, np.transpose(root_cov)), axis=1)

    m = np.concatenate((first_row, other_rows), axis=0)
    r = np.linalg.qr(m, mode="r")
    root_sigma = r[0, 0]
    kalman_gain = r[0, 1:] / root_sigma
    updated_root_cov = np.transpose(np.array(r[1:, 1:]))

    updated_state = state + kalman_gain * residual

    return updated_state, updated_root_cov

@jit(nopython=True)
def numba_update(state, root_cov, measurement, loadings, meas_var):
    """Update *state* and *root_cov with* with a *measurement*.
    Args:
        state (np.ndarray): pre-update estimate of the unobserved state vector
        root_cov (np.ndarray): lower triangular matrix square-root of the
            covariance matrix of the state vector before the update
        measurement (float): the measurement to incorporate
        loadings (np.ndarray): the factor loadings
        meas_var (float): The variance of the incorporated measurement.
    Returns:
        updated_state (np.ndarray)
        updated_root_cov (np.ndarray)
    """
    print('hello')
    expected_measurement = np.dot(state, loadings)
    residual = measurement - expected_measurement
    f_star = np.transpose(np.dot(np.transpose(root_cov), loadings)).reshape(-1, 1)
    first_row = np.array([np.sqrt(meas_var)] + [0] * len(loadings))
    first_row = first_row.reshape((1, -1))


    other_rows = np.concatenate((f_star, np.transpose(root_cov)), axis=1)
    m = np.concatenate((first_row, other_rows), axis=0)
    print(m)
    print('hello')
    r = np.linalg.qr(m)[1]
    root_sigma = r[0, 0]
    kalman_gain = r[0, 1:] / root_sigma
    updated_root_cov = np.transpose(r[1:, 1:])

    updated_state = state + kalman_gain * residual

    return updated_state, updated_root_cov
