import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

sys.path.append("..")


from utils.inference.anonymization import rvMF


def expand(x, y, gap=1e-4):
    add = np.tile([0, gap, np.nan], len(x))
    x1 = np.repeat(x, 3) + add
    y1 = np.repeat(y, 3) + add
    return x1, y1


if __name__ == "__main__":
    plt.rcParams["lines.solid_capstyle"] = "round"

    samples = []

    emb_dim = 128
    pps = [1, 1e1, 1e2, 1e3, 1e4, 1e5]
    colors = ["green", "blue", "red", "yellow", "purple", "gray"]

    real = np.random.rand(emb_dim) - 0.5
    real = real / np.linalg.norm(real)
    samples.append(real)

    # for i in range(1000):
    #     uni = np.random.uniform(low=-1, high=1, size=emb_dim)
    #     uni = uni / np.linalg.norm(uni)
    #     samples.append(uni)

    for pp in pps:
        for i in range(1000):
            eps = rvMF(1, real * pp)
            eps = np.squeeze(eps)
            eps = eps / np.linalg.norm(eps)
            samples.append(eps)

    samples = np.asarray(samples)
    S = TSNE(n_components=2).fit_transform(samples)

    # plt.scatter(S[1:1000, 0], S[1:1000, 1], color="yellow", alpha=1, label="Uniform")

    for i, (pp, color) in enumerate(zip(pps, colors)):
        low = 1000 * (i) + 1
        high = 1000 * (i + 1)

        # using a dumb trick to get non-overlapping transparency in points
        plt.plot(
            *expand(S[low:high, 0], S[low:high, 1]),
            color=color,
            lw=5,
            alpha=0.5,
            label=f"eps={pp:0.0f}",
        )

    plt.scatter(
        S[0, 0], S[0, 1], color="k", alpha=1, label="Real sample", zorder=np.inf
    )

    plt.legend()
    plt.title("TSNE visualization of embeddings sampled from VMF distribution")
    plt.axis("off")
    plt.savefig("vmf_tsne.jpg")
