import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tqdm
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("..")

from utils.inference.anonymization import rvMF

if __name__ == "__main__":
    emb_dim = 512

    spacings = [
        np.linspace(1, 10, 10),
        np.linspace(11, 100, 10),
        np.linspace(101, 1000, 10),
        np.linspace(1001, 10000, 10),
        np.linspace(10001, 100000, 10),
    ]

    fig, ax = plt.subplots(1, 5)

    for j, spacing in enumerate(spacings):
        for eps in tqdm.tqdm(spacing):
            cos_sims = []
            for i in range(100):
                real = np.random.rand(emb_dim) - 0.5
                real = real / np.linalg.norm(real)

                rotated = rvMF(1, np.copy(real) * eps)
                rotated = np.squeeze(rotated)
                rotated = rotated / np.linalg.norm(rotated)

                cs = np.mean(
                    cosine_similarity(real.reshape(1, -1), rotated.reshape(1, -1))
                )
                cos_sims.append(cs)

            ax[j].scatter([eps], [np.mean(cos_sims)], color="blue")
            # ax[j].plot(
            #     [eps, eps],
            #     [
            #         np.mean(cos_sims) - np.std(cos_sims),
            #         np.mean(cos_sims) + np.std(cos_sims),
            #     ],
            #     color="blue",
            # )

            cis = scipy.stats.norm.interval(
                0.95, loc=np.mean(cos_sims), scale=np.std(cos_sims)
            )
            ax[j].plot([eps, eps], [cis[0], cis[1]], color="blue")

            ax[j].set_ylim(-0.2, 1.05)
            if j != 0:
                ax[j].set_ylabel(None)
                ax[j].set_yticks([])
            ax[j].set_xlabel("epsilon")
            ax[j].set_xticks([spacing[0], (spacing[0] + spacing[-1]) // 2, spacing[-1]])

    ax[0].set_ylabel("cosine similarity")
    fig.set_size_inches(12, 4)
    fig.suptitle("Similarity on 512D embeddings resampled from VMF distribution")
    plt.tight_layout()
    plt.savefig("vmf_cos_sim.jpg")
