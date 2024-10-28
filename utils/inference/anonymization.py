import numpy as np
import scipy
import torch
from scipy.stats import uniform_direction

LAST_EMBEDDING = np.zeros(512)
COUNT = 0


def anonymize(embedding_torch, epsilon=1, theta=90):
    global LAST_EMBEDDING, COUNT
    # reshape to a 512 numpy array
    device = embedding_torch.device
    embedding = embedding_torch.cpu().detach().numpy()
    embedding = np.squeeze(embedding)
    real_std = np.std(embedding)
    print(f"real embedding stats: mean={np.mean(embedding)}, std={np.std(embedding)}")

    # verify that the embedding is normalized
    UD = uniform_direction(dim=512)
    embedding = embedding / np.linalg.norm(embedding)

    # first, apply a random rotation sampled from VMF to give DP guarantees
    if epsilon > 0:
        sample = rvMF(1, embedding * epsilon)
        rotated = sample[0]

    elif epsilon == 0:
        rotated = UD.rvs()

    # then apply a follow-up rotation
    if theta > 0:
        theta_rads = theta * np.pi / 180
        rotated = rotate_embedding(rotated, theta_rads)

    rotated = rotated / np.std(rotated)
    rotated = rotated * real_std

    print(f"rotated embedding stats: mean={np.mean(rotated)}, std={np.std(rotated)}")

    print(f"for {COUNT}: {(rotated - LAST_EMBEDDING)[:, 0:10]}")
    COUNT += 1
    LAST_EMBEDDING = rotated

    embedding_torch = torch.from_numpy(rotated)
    embedding_torch = embedding_torch.to(device=device)

    return embedding_torch


def rotate_embedding(embedding: np.ndarray, theta_rads: float) -> np.ndarray:
    # sample a random vector in R^512
    x1 = np.random.uniform(low=-1.0, high=1.0, size=512)
    # generate orthonormal basis between
    orth = scipy.linalg.orth(np.array([x1, embedding]).T)
    # extract our unit basis vectors
    x1 = orth[:, 0]
    x1 = x1.reshape((512, 1))
    x1 = x1 / np.linalg.norm(x1)
    x2 = orth[:, 1]
    x2 = x2.reshape((512, 1))
    x2 = x2 / np.linalg.norm(x2)
    # compute exponential rotation matrix
    e_A = (
        np.identity(512)
        + (np.matmul(x2, x1.T) - np.matmul(x1, x2.T)) * np.sin(theta_rads)
        + (np.matmul(x1, x1.T) + np.matmul(x2, x2.T)) * (np.cos(theta_rads) - 1)
    )

    ## computing the exponential matrix works too but is slower
    # L = np.matmul(x2, x1.T) - np.matmul(x1, x2.T)
    # e_A = scipy.linalg.expm(theta_rads * L)

    # multiply rotation matrix against id embedding
    rotated = np.matmul(e_A, embedding)
    return rotated


# VMF code from https://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python


def sample_tangent_unit(mu):
    mat = np.matrix(mu)

    if mat.shape[1] > mat.shape[0]:
        mat = mat.T

    U, _, _ = scipy.linalg.svd(mat)
    nu = np.matrix(np.random.randn(mat.shape[0])).T
    x = np.dot(U[:, 1:], nu[1:, :])
    return x / scipy.linalg.norm(x)


def rW(n, kappa, m):
    dim = m - 1
    b = dim / (np.sqrt(4 * kappa * kappa + dim * dim) + 2 * kappa)
    x = (1 - b) / (1 + b)
    c = kappa * x + dim * np.log(1 - x * x)

    y = []
    for i in range(0, n):
        done = False
        while not done:
            z = scipy.stats.beta.rvs(dim / 2, dim / 2)
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            u = scipy.stats.uniform.rvs()
            if kappa * w + dim * np.log(1 - x * w) - c >= np.log(u):
                done = True
        y.append(w)
    return y


def rvMF(n, theta):
    dim = len(theta)
    kappa = np.linalg.norm(theta)
    mu = theta / kappa

    result = []
    for sample in range(0, n):
        w = rW(n, kappa, dim)
        w = np.asarray(w)
        print(f"w: {w.shape}")
        # v = np.random.randn(dim)
        # v = v / np.linalg.norm(v)
        v = sample_tangent_unit(mu)
        v = np.squeeze(v)

        print(f"v: {v.shape}")

        result.append(np.sqrt(1 - w**2) * v + w * mu)

    return result
