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

from flooder import flood_complex

from flooder.datasets import CoralDataset, MCBDataset, RocksDataset, SwisscheeseDataset, ModelNet10Dataset


def setup_cmdline_parsing():
    description = """**Flooder options**"""
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
        default="/tmp/coral",
        help="Root folder of CORALS dataset (default: %(default)s)",
    )
    group0.add_argument(
        "--num-landmarks",
        metavar="INT",
        type=int,
        default=2000,
        help="Number of landmarks for Flood complex (default: %(default)s)",
    )
    group0.add_argument(
        "--fpsh",
        metavar="INT",
        type=int,
        default=5,
        help="FPS height (default: %(default)s)",
    )
    group0.add_argument(
        "--points-per-edge",
        metavar="INT",
        type=int,
        default=20,
        help="Points per edge for Flood PH (default: %(default)s)",
    )
    group0.add_argument(
        "--batch-size",
        metavar="INT",
        type=int,
        default=64,
        help="Batch size for Flood complex (default: %(default)s)",
    )
    group0.add_argument(
        "--device",
        metavar="STR",
        type=str,
        default="cuda",
        help="Device used for Flood complex (default: %(default)s)",
    )
    return parser


def main():
    parser = setup_cmdline_parsing()
    args = parser.parse_args()
    print(args)
    assert args.dataset in ['coral', 'rocks', 'mcb', 'swisscheese', 'modelnet10']

    if not os.path.exists(os.path.join(args.root, "floodph")):
        os.makedirs(os.path.join(args.root, "floodph"))
    else:
        print(f'Directory {os.path.join(args.root, "floodph")} exists. Abort')
        exit()

    ph_meta_file = os.path.join(args.root, "floodph", "meta.yaml")

    if args.dataset == 'coral': dataset = CoralDataset(args.root)
    if args.dataset == 'mcb': dataset = MCBDataset(args.root)
    if args.dataset == 'rocks': dataset = RocksDataset(args.root)
    if args.dataset == 'swisscheese': dataset = SwisscheeseDataset(args.root) 
    if args.dataset == 'modelnet10': dataset = ModelNet10Dataset(args.root)

    total_files = len(dataset)

    records = {}
    t00 = time.time()
    for cnt, data in enumerate(dataset):
        label = data.y
        num_points = data.x.shape[0]

        
        out_file = os.path.join(args.root, f"floodph/{data.name}.pt")

        t0 = time.time()
        st = flood_complex(
            data.x.to(args.device),
            args.num_landmarks,
            points_per_edge=args.points_per_edge,
            batch_size=args.batch_size,
            fps_h=args.fpsh,
            use_triton=True,
            return_simplex_tree=True,
            # use_random_landmarks=args.use_random_landmarks,
        )
        ela_complex = time.time() - t0

        t0 = time.time()
        st.compute_persistence()
        ds = [st.persistence_intervals_in_dimension(i) for i in range(0, 3)]
        ela_ph = time.time() - t0
        ela_all = time.time() - t00

        print(
            f"\r[{cnt+1}/{total_files}] Elapsed time {ela_all:.1f}s of approx. {ela_all/(cnt+1)*total_files:.1f}s. Last was in {ela_complex + ela_ph:.2f}s.",
            end="",
        )
        torch.save(ds, out_file)

        records[out_file] = {
            "label": label,
            "complex": "flood",
            "points_per_edge": args.points_per_edge,
            "fpsh": args.fpsh,
            "num_landmarks": args.num_landmarks,
            "ela_complex": ela_complex,
            "ela_ph": ela_ph,
            "gpu": torch.cuda.get_device_name(args.device),
        }

    yaml_data = {"data": records}
    with open(ph_meta_file, "w") as f:
        yaml.dump(yaml_data, f)
    print("")


if __name__ == "__main__":
    main()
