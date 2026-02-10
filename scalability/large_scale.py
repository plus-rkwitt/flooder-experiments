import os
import random
import torch
import numpy as np
from timeit import default_timer as timer
import pandas as pd
import gudhi
import argparse
import yaml
from dataclasses import dataclass

from flooder import flood_complex
from flooder.datasets import LargePointCloudDataset


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda')



def __main__():
    parser = argparse.ArgumentParser(description="Run flooder and alpha complex runtimes.")
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--run_alpha', action='store_true', help="Run alpha complex runtimes.")
    parser.add_argument('--outdir', type=str, default=None, help="Output directory for results.")
    parser.add_argument('--idx', type=int, default=None)
    args = parser.parse_args()
    print(args)
    results = []

    dataset = LargePointCloudDataset(root=args.root)
    print(dataset[args.idx])
    points = dataset[args.idx].x
    print(points.shape)
    points = points.to(device)
    torch.cuda.synchronize()

    startt = timer()
    st = flood_complex(
            points,
            2000,
            points_per_edge=20,
            batch_size=128,
            fps_h=9,
            use_triton=True,
            return_simplex_tree=True,
        )

    torch.cuda.synchronize()
    t1 = timer()-startt
    st.compute_persistence()
    tboth = timer() - startt
    t2 = tboth - t1
    results.append({
        "method": "Flood",
        "tComplex": t1,
        "tPH": t2,
        "tBoth": tboth,
    })
    pdiagram_flood = []
    pdiagram_flood.append(st.persistence_intervals_in_dimension(0))
    pdiagram_flood.append(st.persistence_intervals_in_dimension(1))
    pdiagram_flood.append(st.persistence_intervals_in_dimension(2))

    # Sub alpha
    pdiagram_sub = []
    startt = timer()
    points_sub = points[torch.randperm(points.size(0))[:75000]]
    sub = gudhi.AlphaComplex(points_sub).create_simplex_tree(output_squared_values=False)
    t1 = timer() - startt
    sub.compute_persistence()
    tboth = timer() - startt
    t2 = tboth - t1
    results.append({
        "method": "SubAlpha",
        "tComplex": t1,
        "tPH": t2,
        "tBoth": tboth,
    })
    pdiagram_sub.append(sub.persistence_intervals_in_dimension(0))
    pdiagram_sub.append(sub.persistence_intervals_in_dimension(1))
    pdiagram_sub.append(sub.persistence_intervals_in_dimension(2))

    # Full alpha
    pdiagram_alpha = []
    if args.run_alpha:
        startt = timer()
        alpha = gudhi.AlphaComplex(points).create_simplex_tree(output_squared_values=False)
        t1 = timer()-startt
        alpha.compute_persistence()
        tboth = timer() - startt
        t2 = tboth - t1
        results.append({
            "method": "Alpha",
            "tComplex": t1,
            "tPH": t2,
            "tBoth": tboth,
        })
        pdiagram_alpha.append(alpha.persistence_intervals_in_dimension(0))
        pdiagram_alpha.append(alpha.persistence_intervals_in_dimension(1))
        pdiagram_alpha.append(alpha.persistence_intervals_in_dimension(2))

    df = pd.DataFrame(results)
    print(df)
    torch.save((df, pdiagram_flood, pdiagram_alpha, pdiagram_sub), os.path.join(args.outdir, f"results_{args.idx}.pt"))


if __name__ == "__main__":
    __main__()
