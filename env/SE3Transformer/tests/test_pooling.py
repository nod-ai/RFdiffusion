# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""
Tests to verify that PyG-based pooling gives the same results as DGL pooling.
"""

import dgl
import pytest
import torch

from se3_transformer.model.graph import DGLGraphWrapper, PyTorchGraph
from se3_transformer.model.layers.pooling import GPooling


def _create_batched_graphs(batch_sizes, feat_dim, device='cpu'):
    """
    Create matching DGL and PyTorch graphs with the same structure and features.

    Args:
        batch_sizes: List of node counts per graph in the batch
        feat_dim: Feature dimension for type-0 features
        device: Device to create tensors on

    Returns:
        dgl_graph: DGLGraphWrapper with batched graph
        pyg_graph: PyTorchGraph with same structure
        features: Dict with '0' key containing node features
    """
    # Create individual DGL graphs and batch them
    graphs = []
    for n_nodes in batch_sizes:
        # Create a simple graph where each node connects to the next
        if n_nodes > 1:
            src = torch.arange(n_nodes - 1)
            dst = torch.arange(1, n_nodes)
            g = dgl.graph((src, dst), num_nodes=n_nodes)
        else:
            g = dgl.graph(([], []), num_nodes=n_nodes)
        graphs.append(g)

    batched_dgl = dgl.batch(graphs).to(device)
    batched_dgl.edata['rel_pos'] = torch.zeros(batched_dgl.num_edges(), 3, device=device)

    dgl_graph = DGLGraphWrapper(batched_dgl)

    # Create PyTorchGraph with the same structure
    src, dst = batched_dgl.edges()
    pyg_graph = PyTorchGraph(
        src=src,
        dst=dst,
        num_nodes=batched_dgl.num_nodes(),
        batch_num_nodes=batched_dgl.batch_num_nodes(),
    )
    pyg_graph.edata['rel_pos'] = torch.zeros(batched_dgl.num_edges(), 3, device=device)

    # Create random features - same for both graphs
    total_nodes = sum(batch_sizes)
    # Features shape: [num_nodes, feat_dim, 1] (type-0 features have 1 channel)
    feat = torch.randn(total_nodes, feat_dim, 1, device=device)
    features = {'0': feat}

    return dgl_graph, pyg_graph, features


class TestPoolingEquivalence:
    """Test that PyG pooling matches DGL pooling."""

    @pytest.mark.parametrize("pool_type", ['max', 'avg'])
    @pytest.mark.parametrize("batch_sizes", [
        [10],           # Single graph
        [5, 5],         # Two equal graphs
        [3, 7, 5],      # Three unequal graphs
        [1, 10, 1],     # Mixed with single-node graphs
        [100],          # Larger single graph
        [20, 30, 50],   # Larger batch
    ])
    @pytest.mark.parametrize("feat_dim", [16, 32, 64])
    def test_pooling_equivalence(self, pool_type, batch_sizes, feat_dim):
        """Test that DGL and PyG pooling produce the same results."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        dgl_graph, pyg_graph, features = _create_batched_graphs(
            batch_sizes, feat_dim, device=device
        )

        # Create pooling module
        pooling = GPooling(feat_type=0, pool=pool_type).to(device)

        # Get results from both backends
        dgl_result = pooling(features, dgl_graph)
        pyg_result = pooling(features, pyg_graph)

        # Check shapes match and values are close
        torch.testing.assert_close(dgl_result, pyg_result, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("graph_type", ["dgl", "pyg"])
    def test_max_pooling_correctness(self, graph_type):
        """Test that max pooling actually takes the maximum."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create a simple case where we know the answer
        batch_sizes = [3, 2]
        feat_dim = 4

        dgl_graph, pyg_graph, _ = _create_batched_graphs(
            batch_sizes, feat_dim, device=device
        )
        graph = dgl_graph if graph_type == "dgl" else pyg_graph

        # Create features where we know the max
        # Graph 0: nodes 0,1,2 with values [1,2,3], [4,5,6], [7,8,9], [10,11,12]
        # Graph 1: nodes 3,4 with values [100,200], [300,400], [500,600], [700,800]
        feat = torch.tensor([
            [[1], [4], [7], [10]],      # node 0
            [[2], [5], [8], [11]],      # node 1
            [[3], [6], [9], [12]],      # node 2 (max for graph 0)
            [[100], [300], [500], [700]],  # node 3
            [[200], [400], [600], [800]],  # node 4 (max for graph 1)
        ], dtype=torch.float32, device=device)
        features = {'0': feat}

        pooling = GPooling(feat_type=0, pool='max').to(device)

        result = pooling(features, graph)

        # Expected: max of each graph
        expected = torch.tensor([
            [3, 6, 9, 12],        # max of graph 0
            [200, 400, 600, 800], # max of graph 1
        ], dtype=torch.float32, device=device)

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("graph_type", ["dgl", "pyg"])
    def test_avg_pooling_correctness(self, graph_type):
        """Test that avg pooling actually computes the mean."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create a simple case where we know the answer
        batch_sizes = [2, 3]
        feat_dim = 2

        dgl_graph, pyg_graph, _ = _create_batched_graphs(
            batch_sizes, feat_dim, device=device
        )
        graph = dgl_graph if graph_type == "dgl" else pyg_graph

        # Create features where we know the mean
        # Graph 0: nodes 0,1 -> mean should be (node0 + node1) / 2
        # Graph 1: nodes 2,3,4 -> mean should be (node2 + node3 + node4) / 3
        feat = torch.tensor([
            [[2], [4]],    # node 0: [2, 4]
            [[4], [8]],    # node 1: [4, 8] -> graph 0 mean: [3, 6]
            [[3], [6]],    # node 2: [3, 6]
            [[6], [12]],   # node 3: [6, 12]
            [[9], [18]],   # node 4: [9, 18] -> graph 1 mean: [6, 12]
        ], dtype=torch.float32, device=device)
        features = {'0': feat}

        pooling = GPooling(feat_type=0, pool='avg').to(device)

        result = pooling(features, graph)

        expected = torch.tensor([
            [3, 6],   # mean of graph 0
            [6, 12],  # mean of graph 1
        ], dtype=torch.float32, device=device)

        torch.testing.assert_close(result, expected)
