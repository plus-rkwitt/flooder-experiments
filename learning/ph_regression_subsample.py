import os
import torch
import yaml
import time
import random
import argparse
import warnings
import numpy as np
from pathlib import Path
from flaml import AutoML
from operator import itemgetter
from collections import defaultdict
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter


from flooder.datasets import CoralDataset, MCBDataset, RocksDataset, SwisscheeseDataset, ModelNet10Dataset
from helpers import set_seed, vectorize, parametrize_vectorization

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")

score_fns = {
    "mse": lambda y, yhat: mean_squared_error(y, yhat),
    "mae": lambda y, yhat: mean_absolute_error(y, yhat),
    "r2": lambda y, yhat: r2_score(y,yhat),
}



def setup_cmdline_parsing():
    description = """**LGBM PD Classifier**"""
    parser = argparse.ArgumentParser(
        description=Markdown(description, style="argparse.text"),
        formatter_class=RichHelpFormatter,
    )
    group0 = parser.add_argument_group("Data loading/saving options")
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
        help="Root folder for dataset (default: %(default)s)",
    )
    group0.add_argument(
        "--device",
        metavar="STR",
        type=str,
        default="cpu",
        help="Use cuda:x for GPU vectorization, or cpu (default: %(default)s)",
    )
    group0.add_argument(
        "--splits-file",
        metavar="STR",
        type=str,
        default="splits.yaml",
        help="File with dataset splits (default: %(default)s)",
    )
    group0.add_argument(
        "--phdirs",
        type=str,
        nargs="+",
        default=[],
        help="List of PH directories (default: %(default)s)",
    )
    group0.add_argument(
        "--stats-file",
        metavar="STR",
        type=str,
        default=None,
        help="File with training/testing statistics (default: %(default)s)",
    )
    group1 = parser.add_argument_group("Vectorization options")
    group1.add_argument(
        "--vectorize-bs",
        metavar="STRING",
        type=int,
        dest="vectorize_bs",
        default=-1,
        help="Batchsize for vectorizing large diagrams (default: %(default)s)",
    )
    group0.add_argument(
        "--num-points-for-kmeans",
        metavar="INT",
        type=int,
        default=100000,
        help="Number of points to use for KMeans++ clustering (default: %(default)s)",
    )
    group1.add_argument(
        "--num-structure-elements",
        metavar="INT",
        type=int,
        default=64,
        help="Number of structure elements to use (default: %(default)s)",
    )
    group1.add_argument(
        "--stretch-quantile",
        metavar="FLOAT",
        type=float,
        default=0.05,
        help="Quantile of lifetimes used for selecting stretch parameter for vectorization (default: %(default)s)",
    )
    group2 = parser.add_argument_group("AutoML options")
    group2.add_argument(
        "--flaml-classifier",
        metavar="STR",
        type=str,
        default="lgbm",
        help="Classifier to use for FLAML (default: %(default)s)",
    )
    group2.add_argument(
        "--seed",
        metavar="INT",
        type=int,
        default=42,
        help="Seed the model (default: %(default)s)",
    )
    group2.add_argument(
        "--metric-to-use",
        metavar="STRING",
        type=str,
        dest="metric_to_use",
        default="mse",
        help="Metric to use for AutoML (default: %(default)s)",
    )
    group2.add_argument(
        "--time-budget",
        metavar="INT",
        type=int,
        default=600,
        help="Time budget in sec for AutoML (default: %(default)s)",
    )
    return parser


def main():
    parser = setup_cmdline_parsing()
    args = parser.parse_args()
    set_seed(args.seed)

    w = 30
    print("-" * w)
    print(yaml.dump(vars(args)), end="")
    print("-" * w)

    trn_tracker = defaultdict(list)  # track trn stats
    tst_tracker = defaultdict(list)  # track tst stats
    val_tracker = defaultdict(list)  # track val stats

    if args.dataset == 'rocks': dataset = RocksDataset(args.root)

    split_ids = list(dataset.splits.keys())
    num_runs = len(split_ids)

    dgms, labs = [], []
    for phdir in args.phdirs:
        dgms_dir = []
        for data in dataset:
            dgm = torch.load(os.path.join(phdir, f"{data.name}.pt"), weights_only=False)
            dgms_dir.append(dgm)
        dgms.append(dgms_dir)
    labs = [data.surface for data in dataset]

    for k in split_ids:
        trn_idx = dataset.splits[k]["trn"]
        tst_idx = dataset.splits[k]["tst"]
        val_idx = dataset.splits[k]["val"]

        labs_trn, labs_tst, labs_val = [
            np.array(itemgetter(*idx)(labs))
            for idx in (trn_idx, tst_idx, val_idx)
        ]
        dgms_trn, dgms_tst, dgms_val = [], [], []
        dgms_trn_stacked = []
        for dgms_dir in dgms:
            dgms_dir_trn, dgms_dir_tst, dgms_dir_val = [
                itemgetter(*idx)(dgms_dir) for idx in (trn_idx, tst_idx, val_idx)
            ]
            dgms_trn.append(dgms_dir_trn)
            dgms_tst.append(dgms_dir_tst)
            dgms_val.append(dgms_dir_val)
            dgms_trn_stacked.extend(dgms_dir_trn)

        t0 = time.time()
        fns = parametrize_vectorization(args, dgms=dgms_trn_stacked)
        ela_fit_parameters = time.time() - t0
        print(f"ela fit: {ela_fit_parameters:.1f}")

        trn_v_all, tst_v_all, val_v_all = [], [], []
        for i in range(len(args.phdirs)):
            trn_v, tst_v, val_v = [
                    vectorize(fns, d, args.vectorize_bs, args.device)
                    for d in (dgms_trn[i], dgms_tst[i], dgms_val[i])
                ]
            trn_v_all.append(trn_v)
            tst_v_all.append(tst_v)
            val_v_all.append(val_v)
        trn_v = np.mean(trn_v_all, axis=0)
        tst_v = np.mean(tst_v_all, axis=0)
        val_v = np.mean(val_v_all, axis=0)
        ela_vectorize = time.time() - t0
        print(f"ela vec: {ela_vectorize:.1f}")

        automl = AutoML()
        assert args.metric_to_use in ["mse", "rmse", "mae", "r2"], "Metric not supported"
        settings = {
            "metric": args.metric_to_use,
            "task": "regression",
            "estimator_list": ["lgbm"],
            "seed": args.seed,
            "time_budget": args.time_budget,
            "early_stop": True,
            "verbose": 2,
        }

        automl.fit(
            X_train=trn_v,
            y_train=labs_trn,
            X_val=val_v,
            y_val=labs_val,
            **settings,
        )

        best_model = automl.model
        val_y_hat = best_model.predict(
            val_v
        )  # track validation performance of best model

        X_full = np.concatenate([trn_v, val_v], axis=0)
        y_full = np.concatenate([labs_trn, labs_val], axis=0)

        best_model.fit(X_full, y_full)
        tst_y_hat = best_model.predict(tst_v)
        trn_y_hat = best_model.predict(trn_v)

        trn_str = f"cv-run ({k}/{num_runs}) [{ela_fit_parameters:.2f}s, {ela_vectorize:.2f}s] "
        tst_str = f"cv-run ({k}/{num_runs}) [{ela_fit_parameters:.2f}s, {ela_vectorize:.2f}s] "
        val_str = f"cv-run ({k}/{num_runs}) [{ela_fit_parameters:.2f}s, {ela_vectorize:.2f}s] "

        for score_name, score_fn in score_fns.items():
            trn_score = score_fn(labs_trn, trn_y_hat)
            tst_score = score_fn(labs_tst, tst_y_hat)
            val_score = score_fn(labs_val, val_y_hat)

            trn_tracker[score_name].append(float(trn_score))
            tst_tracker[score_name].append(float(tst_score))
            val_tracker[score_name].append(float(val_score))

            trn_str += f"{score_name}: {trn_score:.4f}, "
            tst_str += f"{score_name}: {tst_score:.4f}, "
            val_str += f"{score_name}: {val_score:.4f}, "

        trn_tracker["labs_trn"].append(labs_trn.tolist())
        trn_tracker["trn_y_hat"].append(trn_y_hat.tolist())
        tst_tracker["labs_tst"].append(labs_tst.tolist())
        tst_tracker["tst_y_hat"].append(tst_y_hat.tolist())
        val_tracker["labs_val"].append(labs_val.tolist())
        val_tracker["val_y_hat"].append(val_y_hat.tolist())

        avg_trn_balanced_accuracy_score = np.mean(
            trn_tracker["mse"]
        )
        avg_tst_balanced_accuracy_score = np.mean(
            tst_tracker["mse"]
        )
        std_trn_balanced_accuracy_score = np.std(trn_tracker["mse"])
        std_tst_balanced_accuracy_score = np.std(tst_tracker["mse"])

        trn_str += f" | {avg_trn_balanced_accuracy_score:.4f} +/- {std_trn_balanced_accuracy_score:.4f}"
        tst_str += f" | {avg_tst_balanced_accuracy_score:.4f} +/- {std_tst_balanced_accuracy_score:.4f}"

        print(tst_str)

        if args.stats_file is not None:
            stats = {
                "trn_tracker": dict(trn_tracker),
                "tst_tracker": dict(tst_tracker),
                "val_tracker": dict(val_tracker),
                "args": vars(args),
            }
            with open(args.stats_file, "w") as f:
                yaml.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
