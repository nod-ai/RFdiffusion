"""
SE3Graph: A graph abstraction for SE3Transformer that supports both DGL and PyTorch backends.

This module provides a unified graph interface that can use either:
1. DGL graphs (default, for exact compatibility with original implementation)
2. Pure PyTorch tensors with PyTorch Geometric ops (better compatibility with torch.compile)

The abstract base class defines the interface, and two subclasses implement
the specific backends.
"""

from abc import ABC, abstractmethod
import os
from typing import Tuple

import dgl
import torch
from torch import Tensor
import torch_geometric

# Global setting for graph backend, read once at import time.
# Set USE_PYTORCH_GRAPH=1 to use PyTorchGraph backend.
# Default is DGL backend for exact compatibility with original implementation.
USE_PYTORCH_GRAPH = os.environ.get('USE_PYTORCH_GRAPH', '0') == '1'


class SE3Graph(ABC):
    """
    Abstract base class for SE3Transformer graph representations.

    This class defines the interface that both DGL and PyTorch backends implement.
    The interface includes:
    - edges() -> (src, dst) tensors
    - num_nodes() -> int
    - batch_num_nodes() -> Tensor of node counts per graph in batch
    - edata property for edge data access
    - copy_e_sum(edge_feats) -> aggregate edge features to destination nodes
    - e_dot_v(edge_feats, node_feats) -> dot product of edge and node features
    - edge_softmax(edge_weights) -> softmax over edges per destination node
    """

    @abstractmethod
    def edges(self) -> Tuple[Tensor, Tensor]:
        """Return (src, dst) tensors for all edges."""
        pass

    @abstractmethod
    def num_nodes(self) -> int:
        """Return the total number of nodes."""
        pass

    @abstractmethod
    def batch_num_nodes(self) -> Tensor:
        """Return tensor of node counts per graph in batch."""
        pass

    @abstractmethod
    def to(self, device, **kwargs) -> 'SE3Graph':
        """Move the graph to the specified device."""
        pass

    @property
    @abstractmethod
    def edata(self):
        """Access edge data (dict-like interface)."""
        pass

    @abstractmethod
    def copy_e_sum(self, edge_feats: Tensor) -> Tensor:
        """
        Sum edge features to destination nodes.

        Args:
            edge_feats: Edge features of shape [num_edges, ...]

        Returns:
            Node features of shape [num_nodes, ...] with edge features summed to destinations
        """
        pass

    @abstractmethod
    def e_dot_v(self, edge_feats: Tensor, node_feats: Tensor) -> Tensor:
        """
        Dot product of edge features with destination node features.

        Args:
            edge_feats: Edge features of shape [num_edges, ...]
            node_feats: Node features of shape [num_nodes, ...]

        Returns:
            Dot product result of shape [num_edges, ..., 1]
        """
        pass

    @abstractmethod
    def edge_softmax(self, edge_weights: Tensor) -> Tensor:
        """
        Softmax over edges grouped by destination node.

        Args:
            edge_weights: Edge weights of shape [num_edges, ...]

        Returns:
            Normalized edge weights of shape [num_edges, ...]
        """
        pass


class DGLGraphWrapper(SE3Graph):
    """
    SE3Graph implementation that wraps a DGL graph.

    This is the default backend that provides exact compatibility with the
    original SE3Transformer implementation. It uses DGL's graph operations
    which are highly optimized but cause graph breaks in torch.compile.
    """

    def __init__(self, dgl_graph: dgl.DGLGraph):
        """
        Wrap an existing DGL graph.
        """
        self._graph = dgl_graph

    def edges(self) -> Tuple[Tensor, Tensor]:
        """Return (src, dst) tensors for all edges."""
        return self._graph.edges()

    def num_nodes(self) -> int:
        """Return the total number of nodes."""
        return self._graph.num_nodes()

    def batch_num_nodes(self) -> Tensor:
        """Return tensor of node counts per graph in batch."""
        return self._graph.batch_num_nodes()

    def to(self, device, **kwargs) -> 'DGLGraphWrapper':
        """Move the graph to the specified device."""
        return DGLGraphWrapper(self._graph.to(device))

    @property
    def edata(self):
        """Access edge data from the underlying DGL graph."""
        return self._graph.edata

    def copy_e_sum(self, edge_feats: Tensor) -> Tensor:
        """Sum edge features to destination nodes using DGL ops."""
        return dgl.ops.copy_e_sum(self._graph, edge_feats)

    def e_dot_v(self, edge_feats: Tensor, node_feats: Tensor) -> Tensor:
        """Dot product using DGL ops."""
        return dgl.ops.e_dot_v(self._graph, edge_feats, node_feats)

    def edge_softmax(self, edge_weights: Tensor) -> Tensor:
        """Edge softmax using DGL ops."""
        return dgl.ops.edge_softmax(self._graph, edge_weights)


class PyTorchGraph(SE3Graph):
    """
    SE3Graph implementation using pure PyTorch tensors with PyG operations.

    This backend has better compatibility with torch.compile because it avoids
    DGL's dlpack-based tensor conversion which torch.compile can't reason about.
    It uses PyTorch Geometric's scatter and softmax operations.

    Note: Due to different accumulation orders in scatter operations, this
    backend may produce slightly different numerical results compared to DGL.
    The outputs are still valid but won't match exactly.
    """

    def __init__(
        self,
        src: Tensor,
        dst: Tensor,
        num_nodes: int,
        batch_num_nodes: Tensor = None,
    ):
        """
        Create a graph from edge indices and relative positions.

        Args:
            src: Source node indices for each edge [num_edges]
            dst: Destination node indices for each edge [num_edges]
            num_nodes: Total number of nodes in the graph
            batch_num_nodes: Node counts per graph in batch [batch_size] (optional)
        """
        self._src = src
        self._dst = dst
        self._num_nodes = num_nodes
        self._edata = {}
        # Default to single graph if not specified
        if batch_num_nodes is None:
            self._batch_num_nodes = torch.tensor([num_nodes], device=src.device)
        else:
            self._batch_num_nodes = batch_num_nodes

    def edges(self) -> Tuple[Tensor, Tensor]:
        """Return (src, dst) tensors for all edges."""
        return self._src, self._dst

    def num_nodes(self) -> int:
        """Return the total number of nodes."""
        return self._num_nodes

    @property
    def edata(self):
        """Access edge data dictionary."""
        return self._edata

    def batch_num_nodes(self) -> Tensor:
        """Return tensor of node counts per graph in batch."""
        return self._batch_num_nodes

    def to(self, device, **kwargs) -> 'PyTorchGraph':
        """Move the graph to the specified device."""

        new_graph = PyTorchGraph(
            self._src.to(device, **kwargs),
            self._dst.to(device, **kwargs),
            self._num_nodes,
            self._batch_num_nodes.to(device, **kwargs),
        )
        new_graph._edata = {k: (v.to(device, **kwargs) if isinstance(v, Tensor) else v) for k, v in self.edata.items()}
        return new_graph

    def copy_e_sum(self, edge_feats: Tensor) -> Tensor:
        """Sum edge features to destination nodes using PyG scatter."""
        return torch_geometric.utils.scatter(edge_feats, self._dst, dim=0, dim_size=self._num_nodes, reduce='sum')

    def e_dot_v(self, edge_feats: Tensor, node_feats: Tensor) -> Tensor:
        """Dot product of edge features with destination node features."""
        return (edge_feats * node_feats[self._dst]).sum(dim=-1, keepdim=True)

    def edge_softmax(self, edge_weights: Tensor) -> Tensor:
        """Softmax over edges grouped by destination node using PyG softmax."""
        return torch_geometric.utils.softmax(edge_weights, self._dst, num_nodes=self._num_nodes)


def create_graph(
    src: Tensor,
    dst: Tensor,
    num_nodes: int,
    device: torch.device = None,
) -> SE3Graph:
    """
    Factory function to create an SE3Graph using either DGL or PyTorch backend.

    Args:
        src: Source node indices for each edge
        dst: Destination node indices for each edge
        num_nodes: Total number of nodes
        device: Device for the graph (only used for DGL backend)

    Returns:
        SE3Graph instance (either DGLGraphWrapper or PyTorchGraph)
    """
    if USE_PYTORCH_GRAPH:
        return PyTorchGraph(src, dst, num_nodes)
    else:
        # Create DGL graph and wrap it
        if device is None:
            device = src.device
        # Checking the node count is a big performance hit
        dgl_graph = dgl.graph((src, dst), num_nodes=num_nodes, node_count_check=False, device=device)
        return DGLGraphWrapper(dgl_graph)


def from_dgl_graph(dgl_graph: dgl.DGLGraph) -> SE3Graph:
    """
    Construct an SE3Graph from an existing DGL graph.

    Depending on the value of USE_PYTORCH_GRAPH, this will either convert the
    DGL graph to a PyTorchGraph or wrap it in a DGLGraphWrapper. This is useful
    for wrapping graphs returned by dgl.batch() or other DGL operations.

    Args:
        dgl_graph: A DGL graph (possibly batched)

    Returns:
        SE3Graph wrapping the graph (either DGLGraphWrapper or PyTorchGraph)
    """
    if USE_PYTORCH_GRAPH:
        src, dst = dgl_graph.edges()
        g = PyTorchGraph(
            src=src,
            dst=dst,
            num_nodes=dgl_graph.num_nodes(),
            batch_num_nodes=dgl_graph.batch_num_nodes(),
        )
        g._edata = {k: v for k, v in dgl_graph.edata.items()}
        return g
    else:
        return DGLGraphWrapper(dgl_graph)
