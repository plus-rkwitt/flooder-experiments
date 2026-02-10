"""Neural Network Baselines"""

import os
from torch_geometric.data.data import BaseData
import yaml
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data

import torch_geometric.data
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.transforms import KNNGraph
from torch_geometric.utils import to_dense_batch


import sklearn
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter

from baselines.pointmlp import pointMLPElite
from baselines.pointnet import PointNet
from baselines.PVT.model.pvt import pvt
import baselines.PVT.modules.provider as provider

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from flooder.datasets import (
    CoralDataset,
    MCBDataset,
    RocksDataset,
    ModelNet10Dataset,
    SwisscheeseDataset,
)


score_fns = {
    "mse": lambda y, yhat: mean_squared_error(y, yhat),
    "mae": lambda y, yhat: mean_absolute_error(y, yhat),
    "r2": lambda y, yhat: r2_score(y,yhat),
}

def setup_cmdline_parsing():
    description = """**PointNet++**"""
    generic_parser = argparse.ArgumentParser(
        description=Markdown(description, style="argparse.text"),
        formatter_class=RichHelpFormatter,
    )
    group0 = generic_parser.add_argument_group("Data loading/saving arguments")
    group0.add_argument(
        "--dataset",
        type=str,
        default="coral",
        help="Dataset name",
    )
    group0.add_argument(
        "--root",
        metavar="FILE",
        type=str,
        default="/tmp/",
        help="Root folder",
    )
    group0.add_argument(
        "--num-points",
        metavar="INT",
        type=int,
        default=2048,
        help="Subsample pcs (default: %(default)s)",
    )
    group0.add_argument(
        "--num-splits",
        metavar="INT",
        type=int,
        default=-1,
        help="Number of splits that are used (default: %(default)s; all splits in split file)",
    )
    group0.add_argument(
        "--stats-file",
        metavar="FILE",
        type=str,
        default=None,
        help="File for logging stats",
    )
    group0.add_argument(
        "--seed",
        metavar="INT",
        type=int,
        default=42,
        help="Seed the model (default: %(default)s)",
    )
    group0.add_argument(
        "--from-corner",
        action="store_true",
        dest="from_corner",
        help="Use closest points to (0,0,0)."
    )
    group1 = generic_parser.add_argument_group("Training arguments")
    group1.add_argument(
        "--model",
        metavar="STR",
        type=str,
        default='pointnet++',
        help='Neural network architecture to be used (pointnet++, pointmlp, pvt), (default: %(default)s)'
        )
    group1.add_argument(
        "--batch-size",
        metavar="INT",
        type=int,
        default=64,
        help="Batch size (default: %(default)s)",
    )
    group1.add_argument(
        "--lr",
        metavar="FLOAT",
        type=float,
        default=1e-2,
        help="Learning rate (default: %(default)s)",
    )
    group1.add_argument(
        "--gradient-clipping",
        action="store_true",
        help="If set, gradient clipping is used.",
    )
    group1.add_argument(
        "--data-augmentation",
        action="store_true",
        help="If set, use data augmentation (random shift, random scaling) is used.",
    )
    group1.add_argument(
        "--no-data-augmentation",
        action="store_false",
        help="If set, use data augmentation (random shift, random scaling) is used.",
    )
    group1.add_argument(
        "--num-epochs",
        metavar="INT",
        type=int,
        default=200,
        help="Number of training epochs (default: %(default)s)",
    )
    group1.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on (default: %(default)s)",
    )
    group1.add_argument(
        "--weight-decay",
        metavar="FLOAT",
        type=float,
        default=1e-4,
        help="Weight decay (default: %(default)s)",
    )
    return generic_parser


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def normalize_pc_batch_to_unit_sphere(pcs):
    pc_centered = pcs - pcs.mean(dim=1, keepdim=True)  # (N, K, 3)
    furthest_dist = pc_centered.norm(dim=2).max(dim=1)[0]  # (N,)
    normalized_pc = pc_centered / furthest_dist[:, None, None]
    return normalized_pc

def normalize_pc_to_unit_sphere(pc):
    pc_centered = pc - pc.mean(dim=0, keepdim=True)  # (K, 3)
    furthest_dist = pc_centered.norm(dim=1).max(dim=0)[0] 
    normalized_pc = pc_centered / furthest_dist
    return normalized_pc

def subsample_pc(pc, num_points):
    indices = np.random.choice(pc.shape[0], num_points, replace=False)
    return pc[indices]


class ProcessPcsTransform(T.BaseTransform):
    def __init__(self, num_points, knn, normalize=False, from_corner=False):
        self.num_points = num_points
        self.knn = knn
        self.normalize = normalize
        self.from_corner = from_corner

    def __call__(self, data):
        x = data.x
        if self.from_corner:
            assert x.min() >= 0., "Can only use --from-corner if data is positive"
            i = 0
            lower = 10 if self.num_points < 5120 else 20
            for i in range(lower,256):
                indices = (x < i).all(axis=1)
                if indices.sum() > self.num_points:
                    break
            x = x[indices]
            
        if self.normalize:
            x = normalize_pc_to_unit_sphere(x)
        x = subsample_pc(x, self.num_points)

        edge_index = None
        if self.knn > 0:
            edge_index = KNNGraph(k=self.knn)(torch_geometric.data.Data(pos=x)).edge_index
            
        return torch_geometric.data.Data(
            pos=x,
            y=data.surface,
            edge_index=edge_index,
        )


def trn_epoch(args, model, dl, optimizer, criterion, clip=False, augment=False):
    model.train()

    total_loss = 0
    preds = []
    truth = []

    for data in dl:
        label = data.y
        if augment:
            shape = data.pos.shape
            pos = to_dense_batch(data.pos, data.batch)[0]
            pos = pos.numpy()
            # pos = provider.random_point_dropout(pos)  # commented out because unclear how pointnet++ knn graph is affected
            pos[:, :, 0:3] = provider.random_scale_point_cloud(pos[:, :, 0:3])
            pos[:, :, 0:3] = provider.shift_point_cloud(pos[:, :, 0:3])
            data.pos = torch.Tensor(pos).view(shape)
        data, label = data.to(args.device), label.to(args.device)

        optimizer.zero_grad()
        pred = model(data).squeeze()
        loss = criterion(pred, label)

        loss.backward()
        if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        total_loss += loss.item()
        preds.append(pred.cpu().detach().numpy())
        truth.append(data.y.cpu().detach().numpy())      
    return total_loss / len(dl.dataset), preds, truth


@torch.no_grad()
def tst_epoch(args, model, dl, criterion):
    model.eval()

    total_loss = 0
    preds = []
    truth = []

    for data in dl:
        label = data.y
        data, label = data.to(args.device), label.to(args.device)
        pred = model(data).squeeze()
        loss = criterion(pred, label).item()
        preds.append(pred.cpu().numpy())
        truth.append(data.y.cpu().numpy())
        total_loss += loss 
    return total_loss / len(dl.dataset), preds, truth



def main():
    parser = setup_cmdline_parsing()
    args = parser.parse_args()
    set_seed(args.seed)
    w = 30
    print("-" * w)
    print(yaml.dump(vars(args)), end="")
    print("-" * w)
    assert args.dataset in ['coral', 'rocks', 'mcb', 'swisscheese', 'modelnet10']

    supported_models = {
    'pointnet++': lambda num_classes: PointNet(h_dim=512, num_classes=num_classes),
    'pointmlp': lambda num_classes: pointMLPElite(points=args.num_points, num_classes=num_classes),
    'pvt': lambda num_classes: pvt(num_classes=num_classes, in_channels=3),
    }    

    torch.cuda.set_device(args.device)

    trn_tracker = defaultdict(list)
    tst_tracker = defaultdict(list)
    val_tracker = defaultdict(list)

    k = 6 if args.model == 'pointnet++' else 0
    print("-" * w)

    transform = ProcessPcsTransform(args.num_points, k, from_corner=args.from_corner)
    if args.dataset == 'coral': dataset = CoralDataset(args.root, fixed_transform=transform)
    if args.dataset == 'mcb': dataset = MCBDataset(args.root, fixed_transform=transform)
    if args.dataset == 'rocks': dataset = RocksDataset(args.root, fixed_transform=transform)
    if args.dataset == 'swisscheese': dataset = SwisscheeseDataset(args.root, fixed_transform=transform)
    if args.dataset == 'modelnet10': dataset = ModelNet10Dataset(args.root, fixed_transform=transform)
    

    print(dataset)
    print(dataset[0])
    split_ids = list(dataset.splits.keys())

    num_splits = args.num_splits if args.num_splits >= 0 else len(split_ids)
    for split_id in split_ids[:num_splits]:
        ds_trn, ds_val, ds_tst = (
            dataset[ dataset.splits[split_id][split] ] for split in ('trn', 'val', 'tst')
        )
        print(ds_trn, ds_val, ds_tst)
        dl_trn = DataLoader(
            ds_trn, batch_size=args.batch_size, shuffle=True, pin_memory=True
        )
        dl_tst = DataLoader(
            ds_tst, batch_size=args.batch_size, shuffle=False, pin_memory=True
        )
        dl_val = DataLoader(
            ds_val, batch_size=args.batch_size, shuffle=False, pin_memory=True
        )


        num_classes = 1
        model = supported_models[args.model](num_classes).to(args.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.num_epochs
        )

        criterion = nn.MSELoss(reduction='sum')

        best_sk_mse_val = np.inf
        best_sk_mse_tst = np.inf
        best_epoch = 0
        for epoch in range(1, args.num_epochs + 1):
            loss, preds_trn, truth_trn = trn_epoch(args, model, dl_trn, optimizer, criterion, augment=args.data_augmentation, clip=args.gradient_clipping)
            _, preds_val, truth_val = tst_epoch(args, model, dl_val, criterion)
            scheduler.step()
            
            preds_trn = np.concatenate(preds_trn, axis=0)
            truth_trn = np.concatenate(truth_trn, axis=0)
            preds_val = np.concatenate(preds_val, axis=0)
            truth_val = np.concatenate(truth_val, axis=0)

            sk_mse_trn = sklearn.metrics.mean_squared_error(
                truth_trn, preds_trn
            )
            sk_mse_val = sklearn.metrics.mean_squared_error(
                truth_val, preds_val
            )
            
            if sk_mse_val < best_sk_mse_val:
                _, preds_tst, truth_tst = tst_epoch(args, model, dl_tst, criterion)
                preds_tst = np.concatenate(preds_tst, axis=0)
                truth_tst = np.concatenate(truth_tst, axis=0)
                sk_mse_tst = sklearn.metrics.mean_squared_error(
                    truth_tst, preds_tst
                )
                best_sk_mse_val = sk_mse_val
                best_sk_mse_tst = sk_mse_tst
                best_epoch = epoch

                best_preds_trn = preds_trn.copy()
                best_truth_trn = truth_trn.copy()
                best_preds_val = preds_val.copy()
                best_truth_val = truth_val.copy()
                best_preds_tst = preds_tst.copy()
                best_truth_tst = truth_tst.copy()
                
            print("\r", end='')
            print(
                f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Train MSE: {sk_mse_trn:.4f}, Val MSE: {sk_mse_val:.4f}| Best Test-from-Val Acc: {best_sk_mse_tst:.4f}  (epoch {best_epoch})",
                end="",
                flush=True,
            )

        for score_name, score_fn in score_fns.items():
            trn_score = score_fn(best_truth_trn, best_preds_trn)
            val_score = score_fn(best_truth_val, best_preds_val)
            tst_score = score_fn(best_truth_tst, best_preds_tst)  
            trn_tracker[score_name].append(float(trn_score))
            val_tracker[score_name].append(float(val_score))
            tst_tracker[score_name].append(float(tst_score))

        current_mse_accuracy_trn = trn_tracker["mse"][-1]
        current_mse_accuracy_tst = tst_tracker["mse"][-1]
        current_mse_accuracy_tst_avg = np.mean(
            tst_tracker["mse"]
        )
        current_mse_accuracy_tst_std = np.std(
            tst_tracker["mse"]
        )
        print(
            f"{split_id} | Train Acc (balanced): {current_mse_accuracy_trn:.4f},  Test Acc (balanced): {current_mse_accuracy_tst:.4f} | {current_mse_accuracy_tst_avg:.4f} +/- {current_mse_accuracy_tst_std:.4f}"
        )

    if args.stats_file is not None:
        stats = {
            "trn_tracker": dict(trn_tracker),
            "val_tracker": dict(val_tracker),
            "tst_tracker": dict(tst_tracker),
            "args": vars(args),
        }
        with open(args.stats_file, "w") as f:
            yaml.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()