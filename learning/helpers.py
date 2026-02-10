import random
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from torchph.nn.slayer import LinearRationalStretchedBirthLifeTimeCoordinateTransform
from torchph.nn import SLayerExponential

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def vectorize(vec_fns, dgms, batch_size=-1, device="cpu"):
    n_dgms = len(dgms)
    if batch_size == -1:
        batch_size = n_dgms
    n_dims = len(dgms[0])

    dim_vecs = []
    for d in range(n_dims):
        tf, sl = vec_fns[d]
        sl = sl.to(device)
        batches_out = []

        for b in range(0, n_dgms, batch_size):
            batch_slice = dgms[b : b + batch_size]
            inp = [
                tf(torch.as_tensor(sample[d], dtype=torch.float32, device=device))
                for sample in batch_slice
            ]
            with torch.no_grad():
                out = sl(inp)
            batches_out.append(out)
        dim_vec = torch.cat(batches_out, dim=0).cpu()
        dim_vecs.append(dim_vec)
    return torch.cat(dim_vecs, dim=1).numpy()


def parametrize_vectorization(args, dgms):
    dgms_stacked = [
        np.vstack([d[0][:-1] for d in dgms]),
        np.vstack([d[1] for d in dgms]),
        np.vstack([d[2] for d in dgms]),
    ]
    vec_fns = []
    for i in range(0, len(dgms_stacked)):
        T = torch.tensor(dgms_stacked[i], dtype=torch.float32)
        if T.shape[0] < args.num_structure_elements:
            T = torch.cat(
                [T, torch.zeros((args.num_structure_elements - T.shape[0], 2))], dim=0
            )
        print(T.shape)
        selection = torch.randperm(T.shape[0])[0 : args.num_points_for_kmeans]
        T = T[selection]
        T[:, 1] = T[:, 1] - T[:, 0]

        km = KMeans(
            n_clusters=args.num_structure_elements,
            init="k-means++",
            random_state=args.seed,
        )
        km.fit(T)

        ci = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(ci)
        dists, _ = neigh.kneighbors(ci, 2, return_distance=True)
        si = (
            torch.tensor(dists[:, -1] / 2.0, dtype=torch.float32)
            .view(args.num_structure_elements, 1)
            .repeat(1, 2)
        )
        stretch = torch.quantile(T[:, 1], args.stretch_quantile)
        tf = LinearRationalStretchedBirthLifeTimeCoordinateTransform(stretch)
        sl = SLayerExponential(
            args.num_structure_elements, 2, centers_init=ci, sharpness_init=1.0 / si
        )
        vec_fns.append((tf, sl))  # vectorization functions (transform+SLayer) per dim
    return vec_fns