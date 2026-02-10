import os
import yaml
import torch
import time
import argparse
import numpy as np
from uuid import uuid4
from pathlib import Path
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter
from gudhi import AlphaComplex

from flooder.datasets import CoralDataset, MCBDataset, RocksDataset, SwisscheeseDataset, ModelNet10Dataset


def setup_cmdline_parsing():
    description = """**Alpha PH options**"""
    parser = argparse.ArgumentParser(
        description=Markdown(description, style="argparse.text"),
        formatter_class=RichHelpFormatter,
    )
    group0 = parser.add_argument_group("Processing options")
    group0.add_argument(
        "--dataset",
        type=str,
        default="coral",
        help="Dataset name",
    )
    group0.add_argument(
        "--root",
        metavar="FOLDER",
        type=str,
        default=None,
        help="Root folder of CORALS dataset (default: %(default)s)",
    )
    group0.add_argument(
        "--num-points",
        metavar="INT",
        type=int,
        default=None,
        help="Number of points to use for Alpha complex (default: %(default)s)",
    )
    group0.add_argument(
        "--scale-num-points",
        type=float,
        default=None,
        help="Fraction of points to use for Alpha complex (default: %(default)s)",
    )
    group0.add_argument(
        "--seed",
        metavar="Seed",
        type=int,
        default=0,
        help="Random seed for subsampling the points (default: %(default)s)",
    )
    return parser


def main():
    parser = setup_cmdline_parsing()
    args = parser.parse_args()
    print(yaml.dump(vars(args)), end="")

    seed = None if args.seed == -1 else args.seed
    rng = np.random.default_rng(seed=seed)

    assert args.dataset in ['coral', 'rocks', 'mcb', 'swisscheese', 'modelnet10']
    if args.dataset == 'coral': dataset = CoralDataset(args.root)
    if args.dataset == 'mcb': dataset = MCBDataset(args.root)
    if args.dataset == 'rocks': dataset = RocksDataset(args.root)
    if args.dataset == 'swisscheese': dataset = SwisscheeseDataset(args.root) 
    if args.dataset == 'modelnet10': dataset = ModelNet10Dataset(args.root)
    total_files = len(dataset)

    if args.scale_num_points is not None:
        assert (
            args.num_points is None
        ), "Cannot use --scale and --num-points together."
        outdir = os.path.join(args.root, f"alphaph_{args.scale_num_points}_{args.seed}")
    else:
        assert (
            args.num_points is not None
        ), "One of --scale and --num-points must be set."
        outdir = os.path.join(args.root, f"alphaph_{args.num_points}_{args.seed}")
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        print('Folder exists')
        exit()
    print(f"Saving diagrams to {outdir}")
    ph_meta_file = os.path.join(outdir, f"meta.yaml")    

    records = {}
    t00 = time.time()
    for cnt, data in enumerate(dataset):
        out_file = os.path.join(outdir, f"{data.name}.pt")
        label = data.y
        pc = data.x

        if args.scale_num_points is not None:
            sample_n_points = int(float(pc.shape[0]) * float(args.scale_num_points))
        else:
            sample_n_points = args.num_points
        if sample_n_points > pc.shape[0]:
            raise ValueError(
                f"Cannot sample {sample_n_points} points from {pc.shape[0]} without replacement."
            )
        
        indices = rng.choice(pc.shape[0], sample_n_points, replace=False)
        pc_sub = pc[indices]        

        t0 = time.time()
        alpha = AlphaComplex(pc_sub).create_simplex_tree(
            output_squared_values=False
            )
        ela_complex = time.time() - t0

        t0 = time.time()
        alpha.compute_persistence()
        ds = [
            alpha.persistence_intervals_in_dimension(i)
            for i in range(0, pc.shape[1])
        ]
        ela_ph = time.time() - t0
        ela_all = time.time() - t00

        print(
            f"\r[{cnt+1}/{total_files}] Elapsed time {ela_all:.1f}s of approx. {ela_all/(cnt+1)*total_files:.1f}s. Last was in {ela_complex + ela_ph:.2f}s.",
            end="",
        )
        torch.save(ds, out_file)

        records[out_file] = {
            "label": label,
            "complex": "alpha",
            "num_points": args.num_points,
            "scale_num_points": args.scale_num_points,
            "ela_complex": ela_complex,
            "ela_ph": ela_ph,
            "seed": seed,
        }

    yaml_data = {"data": records}
    with open(ph_meta_file, "w") as f:
        yaml.dump(yaml_data, f)
    print("")


if __name__ == "__main__":
    main()
