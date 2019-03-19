from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from MulticoreTSNE import MulticoreTSNE


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    # X = X / 255.0

    print(X.shape)

    tsne = MulticoreTSNE(
        verbose=1,
        n_components=2,
        learning_rate=200,
        n_iter=1000,
        n_iter_without_progress=200,
        perplexity=20,
        min_grad_norm=1e-5,
    )

    Z = tsne.fit_transform(X)
    plt.figure(figsize=(6, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=y, alpha=0.5)
    plt.savefig("./temp.png")
    plt.close()
