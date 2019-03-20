# test modified version of MulticoreTSNE

from MulticoreTSNE import MulticoreTSNE as TSNE

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
from matplotlib import pyplot as plt

import multiprocessing


print(TSNE.__version__)
ncpu = multiprocessing.cpu_count()
ncpu_used = int(ncpu * 0.75)


X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)

tsne = TSNE(
    n_jobs=ncpu_used,
    n_iter_without_progress=100,
    min_grad_norm=1e-04,
    perplexity=65,
    verbose=1,
)
Z = tsne.fit_transform(X)

print("KL loss", tsne.kl_divergence_)
progress_errors = tsne.progress_errors_
progress_errors = progress_errors[np.where(progress_errors > 0)]
print("Loss by iter", progress_errors)

plt.figure(figsize=(5, 2))
plt.plot(progress_errors)
plt.savefig("temp_test_installed_loss.png")

error_per_point = tsne.error_per_point_
sizes = (
    MinMaxScaler(feature_range=(32, 160))
    .fit_transform(error_per_point.reshape(-1, 1))
    .reshape(1, -1)
)

plt.figure(figsize=(6, 6))
plt.scatter(Z[:, 0], Z[:, 1], c=y, s=sizes, alpha=0.4, cmap="jet")
plt.savefig("temp_test_installed_scatter.png")
