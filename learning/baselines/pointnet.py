import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import MessagePassing


# region PointNet++
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr="max")

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(
        self,
        h_j: torch.Tensor,
        pos_j: torch.Tensor,
        pos_i: torch.Tensor,
    ) -> torch.Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]

        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)


class PointNet(torch.nn.Module):
    def __init__(self, h_dim: int = 32, num_classes=10):
        super().__init__()

        self.conv1 = PointNetLayer(3, h_dim)
        self.conv2 = PointNetLayer(h_dim, h_dim)
        self.classifier = nn.Linear(h_dim, num_classes)

    def forward(
        self,
        data
    ) -> torch.Tensor:
        # Perform two-layers of message passing:
        h = self.conv1(h=data.pos, pos=data.pos, edge_index=data.edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=data.pos, edge_index=data.edge_index)
        h = h.relu()
        # Global Pooling:
        h = global_max_pool(h, data.batch)  # [num_examples, hidden_channels]
        return self.classifier(h)


class PHPointNet(torch.nn.Module):
    def __init__(self, h_dim: int = 32, ph_dim=64, num_classes=10):
        super().__init__()

        self.conv1 = PointNetLayer(3, h_dim)
        self.conv2 = PointNetLayer(h_dim, h_dim)
        self.classifier = nn.Linear(h_dim + ph_dim, num_classes)

    def forward(
        self,
        pos: torch.Tensor,
        ph: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:

        # Perform two-layers of message passing:
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        # Global Pooling:
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        h = torch.cat( [h, ph] , dim=-1)
        return self.classifier(h)
