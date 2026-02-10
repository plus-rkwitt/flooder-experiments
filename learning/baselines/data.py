import os
from torch_geometric.data.data import BaseData
import yaml
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn

from torch_geometric.data import Data, Dataset, Batch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import KNNGraph

import sklearn
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from utils.models import PointNet
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter

class SplitPointCloudDataset(Dataset):
    def __init__(
        self,
        root_dir,
        yaml_path,
        split="trn",
        split_id=1,
        num_points=1024,
        k=6,
        normalize_within_sphere=False,
    ):
        super().__init__()
        with open(yaml_path, "r") as f:
            ydata = yaml.safe_load(f)

        self.root_dir = root_dir
        self.transform = None
        self.rel_paths = ydata["rel_paths"]
        self.split_indices = ydata["splits"][split_id][split]

        self.pcs = []
        self.labels = []

        class_names = sorted({p.split("/")[0] for p in self.rel_paths})
        self.class_to_label = {cls_name: i for i, cls_name in enumerate(class_names)}

        for idx in self.split_indices:
            rel_path = self.rel_paths[idx]
            label = self.class_to_label[rel_path.split("/")[0]]
            full_path = os.path.join(self.root_dir, rel_path)
            pc = np.load(full_path).astype(np.float32)
            indices = np.random.choice(pc.shape[0], num_points, replace=False)
            self.pcs.append(pc[indices])
            self.labels.append(label)

        if normalize_within_sphere:
            self.pcs_pt = normalize_pc_batch_to_unit_sphere(
                torch.tensor(np.stack(self.pcs, axis=0), dtype=torch.float32)
            )
        else:
            self.pcs_pt = torch.tensor(np.stack(self.pcs, axis=0), dtype=torch.float32)
        self.labels_pt = torch.tensor(self.labels, dtype=torch.long)

        self.data_objs = [
            Data(
                pos=self.pcs_pt[i],
                y=self.labels[i],
                edge_index=KNNGraph(k=k)(Data(pos=self.pcs_pt[i])).edge_index,
            )
            for i in range(len(self.pcs_pt))
        ]

    def len(self):
        return self.pcs_pt.shape[0]

    def get(self, idx):
        return self.data_objs[idx]
        
    
    def __getitem__(self, idx):
        out =  super().__getitem__(idx)
        return out, out.y