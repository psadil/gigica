import time
import h5py
import numpy as np

from gigica._algorithm import gig_ica_fit_rust

print("loading")
with h5py.File(
    "/Users/psadil/git/manuscripts/denoise-comparison/derivatives/tmp/bigger.mat", "r"
) as f:
    X = np.array(f["data"]).T
    R = np.array(f["ref_data"]).T

print(f"X shape: {X.shape}")
print(f"R shape: {R.shape}")

print("starting run")

start_rust = time.time()
y = gig_ica_fit_rust(X, R, max_iter=100, alpha=0.5, whiten=True, tol=1e-5)
end_rust = time.time()
print(f"{(end_rust - start_rust)=}")
