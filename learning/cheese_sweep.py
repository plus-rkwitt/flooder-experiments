import numpy as np
import os
import yaml
import subprocess
import argparse
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter
import pandas as pd

from flooder.datasets import SwisscheeseDataset
from helpers import set_seed


def setup_cmdline_parsing():
    description = """**Flooder options**"""
    parser = argparse.ArgumentParser(
        description=Markdown(description, style="argparse.text"),
        formatter_class=RichHelpFormatter,
    )
    group0 = parser.add_argument_group("Processing options")
    group0.add_argument(
        "--root",
        type=str,
        default=None,
    )
    group0.add_argument(
        "--output",
        type=str,
        default=None,
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
        default=512,
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
    set_seed(42)


    subprocess.run([
        "python", "leanring/ph_flood.py",
        "--root", args.root,
        "--dataset", "swisscheese",
    ])

    subprocess.run([
        "python", "learning/ph_classify.py",
        "--root", args.root,
        "--dataset", "swisscheese",
        "--phdir", os.path.join(args.root, "floodph"),
        "--stats-file", os.path.join(args.output, "flood_swisscheese_sweep.yaml"),
        "--time-budget", "300",
    ])

    results = yaml.safe_load(open(os.path.join(args.output, "flood_swisscheese_sweep.yaml"), "r"))
    flood_acc = np.mean(results["tst_tracker"]["balanced_accuracy_score"])
    results = yaml.safe_load(open(os.path.join(args.root, "floodph", "meta.yaml"), "r"))
    df = pd.DataFrame(results["data"]).transpose()
    flood_time = df.ela_complex.mean() + df.ela_ph.mean()


    alpha_accs = []
    alpha_stds = []
    alpha_times = []
    Ns =  (10.**np.linspace(3, 5, 9)).tolist() + [1_000_000]
    Ns = [1000, 3000, 10000]
    for N in Ns:
        subprocess.run([
            "python", "ph_alpha.py",
            "--root", args.root,
            "--dataset", "swisscheese",
            "--num-points", f'{int(N)}', 
        ])

        subprocess.run([
            "python", "ph_classify.py",
            "--root", args.root,
            "--dataset", "swisscheese",
            "--phdir", os.path.join(args.root, f"alphaph_{int(N)}_0"),
            "--stats-file", os.path.join(args.output, f"alpha{int(N)}_swisscheese_sweep.yaml"),
            "--time-budget", "300",
        ])

        results = yaml.safe_load(open(os.path.join(args.output, f"alpha{int(N)}_swisscheese_sweep.yaml"), "r"))
        alpha_accs.append( 
            np.mean(results["tst_tracker"]["balanced_accuracy_score"]) 
        )
        alpha_stds.append( 
            np.std(results["tst_tracker"]["balanced_accuracy_score"]) 
        )
        results = yaml.safe_load(open(os.path.join(args.root, f"alphaph_{int(N)}_0", "meta.yaml"), "r"))
        df = pd.DataFrame(results["data"]).transpose()
        alpha_times.append( 
            df.ela_complex.mean() + df.ela_ph.mean() 
        )

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(Ns, alpha_accs, label="Alpha", color="red")
    ax[0].errorbar(Ns, alpha_accs, yerr=alpha_stds, fmt='o', color="red", capsize=5)
    ax[0].set_xscale('log')
    ax[0].set_xlabel("Number of points in Alpha complex")
    ax[0].set_ylabel("Accuracy")
    ax[0].axhline(flood_acc, color='black', linestyle='--')

    ax[1].scatter(Ns, alpha_times, label="Alpha", color="red")
    ax[1].set_xscale('log')
    ax[1].set_xlabel("Number of points in Alpha complex")
    ax[1].set_ylabel("Runtime (s)")
    ax[1].axhline(flood_time, color='black', linestyle='--')

    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "figure_cheese_sweep.png"), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()