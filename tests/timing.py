from time import time
import sys
import numpy as np
import pandas as pd
import kalman
from update import fast_batch_update
from update import pandas_batch_update
import pdb

# load and prepare data
data = pd.read_stata("../chs_data.dta")
data.replace(-100, np.nan, inplace=True)
data = data.query("age == 0")
data.reset_index(inplace=True)
data = data["weightbirth"]
data.fillna(data.mean(), inplace=True)

#print(data.shape)

# fix dimensions
nobs = len(data)
state_names = ["cog", "noncog", "mother_cog", "mother_noncog", "investments"]
nstates = len(state_names)


# construct initial states
states_np = np.zeros((nobs, nstates))
states_pd = pd.DataFrame(data=states_np, columns=state_names)

# construct initial covariance matrices
root_cov = np.linalg.cholesky(
    [
        [0.1777, -0.0204, 0.0182, 0.0050, 0.0000],
        [-0.0204, 0.2002, 0.0592, 0.0261, 0.0000],
        [0.0182, 0.0592, 0.5781, 0.0862, -0.0340],
        [0.0050, 0.0261, 0.0862, 0.0667, -0.0211],
        [0.0000, 0.0000, -0.0340, -0.0211, 0.0087],
    ]
)

root_covs_np = np.zeros((nobs, nstates, nstates))
root_covs_np[:] = root_cov

root_covs_pd = []
for _ in range(nobs):
    root_covs_pd.append(
        pd.DataFrame(data=root_cov, columns=state_names, index=state_names)
    )

# construct measurements
meas_bwght_np = data.values
meas_bwght_pd = data

# construct loadings
loadings_bwght_np = np.array([1.0, 0, 0, 0, 0])
loadings_bwght_pd = pd.Series(loadings_bwght_np, index=state_names)

# construct the variance
meas_var_bwght = 0.8

#pdb.set_trace()
#time the function
runtimes = []
for _i in range(3):
    start = time()
    pandas_batch_update(
        states=states_pd,
        root_covs=root_covs_pd,
        measurements=meas_bwght_pd,
        loadings=loadings_bwght_pd,
        meas_var=meas_var_bwght,
    )
    stop = time()
    runtimes.append(stop - start)

# exclude first run in case you use numba
mean_runtime = np.mean(runtimes[1:])
print("pandas_batch_update took {} seconds.".format(mean_runtime))

# do the same for the fast version
runtimes = []
for _ in range(100):
    start = time()
    kalman.fast_batch_update(
        states=states_np,
        root_covs=root_covs_np,
        measurements=meas_bwght_np.reshape(-1,1),
        loadings=loadings_bwght_np.reshape(5,1),
        meas_var=meas_var_bwght,
    )
    stop = time()
    runtimes.append(stop - start)

# exclude first run in case you use numba
mean_runtime_fast = np.mean(runtimes)
print("fast_batch_update took {} seconds.".format(mean_runtime_fast))
print("This is a speedup of factor {}.".format(mean_runtime / mean_runtime_fast))
