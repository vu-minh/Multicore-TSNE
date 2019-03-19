from functools import partial
from time import time

from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
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
        perplexity=20,
        min_grad_norm=1e-4,
    )
    Z = tsne.fit_transform(X)

    running_time = time() - start_time
    kl_loss = tsne.kl_divergence_
    run_iter = tsne.n_iter_
    print(f"kl_loss={kl_loss} in {run_iter} iters, {running_time} seconds\n")

    plt.figure(figsize=(6, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=y, alpha=0.5)
    plt.savefig(out_name)
    plt.close()


if __name__ == "__main__":
    # X, y = load_iris(return_X_y=True)
    # X = StandardScaler().fit_transform(X)

    X, y = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    # X = X / 255.0

    verbose = 1
    test_tsne(X, TSNE=partial(MulticoreTSNE, n_jobs=2), out_name="multicore.png")
    test_tsne(X, TSNE=SKTSNE, out_name="sklearn.png")
