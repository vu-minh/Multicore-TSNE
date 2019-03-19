from functools import partial
from time import time

import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt

from MulticoreTSNE import MulticoreTSNE
from sklearn.manifold import TSNE as SKTSNE


def test_tsne(X, TSNE, out_name="temp.png"):
    print(f"\nOutput result in {out_name}")
    start_time = time()
    tsne = TSNE(
        verbose=verbose,
        n_components=2,
        learning_rate=200,
        n_iter=1000,
        n_iter_without_progress=150,
        perplexity=10,
        min_grad_norm=1e-4,
    )
    Z = tsne.fit_transform(X)

    running_time = time() - start_time
    kl_loss = tsne.kl_divergence_
    run_iter = tsne.n_iter_
    print(f"kl_loss={kl_loss} in {run_iter} iters, {running_time} seconds\n")

    try:
        print("Progress errors: ")
        progress_errors = tsne.progress_errors_
        progress_errors = progress_errors[np.where(progress_errors > 0)]
        print(progress_errors)
    except AttributeError:
        print("`progress_errors_` is not an attribute of TSNE object\n")

    error_per_point = None
    try:
        print("Get error for each point: ")
        error_per_point = tsne.error_per_point_
        print("Original error: \n")
        print(error_per_point)
        error_per_point = (
            MinMaxScaler(feature_range=(32, 160))
            .fit_transform(error_per_point.reshape(-1, 1))
            .reshape(1, -1)
        )
        print(error_per_point)
    except AttributeError:
        print("`error_per_point_` is not an attribute of TSNE object\n")

    plt.figure(figsize=(6, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=y, s=error_per_point, alpha=0.5, cmap="jet")
    plt.savefig(out_name)
    plt.close()


if __name__ == "__main__":
    TEST_SMALL = False
    verbose = 1

    if TEST_SMALL:
        X, y = load_iris(return_X_y=True)
    else:
        X, y = load_digits(return_X_y=True)
        # X = X / 255.0
    X = StandardScaler().fit_transform(X)

    test_tsne(X, TSNE=partial(MulticoreTSNE, n_jobs=2), out_name="multicore.png")
    test_tsne(X, TSNE=SKTSNE, out_name="sklearn.png")
