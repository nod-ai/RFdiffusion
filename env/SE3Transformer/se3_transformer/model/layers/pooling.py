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

from typing import Dict, Literal

import torch
import torch.nn as nn
from dgl.nn.pytorch import AvgPooling, MaxPooling
from torch import Tensor
from torch_geometric.nn import global_max_pool, global_mean_pool

from se3_transformer.model.graph import SE3Graph, DGLGraphWrapper


class GPooling(nn.Module):
    """
    Graph max/average pooling on a given feature type.
    The average can be taken for any feature type, and equivariance will be maintained.
    The maximum can only be taken for invariant features (type 0).
    If you want max-pooling for type > 0 features, look into Vector Neurons.
    """

    def __init__(self, feat_type: int = 0, pool: Literal['max', 'avg'] = 'max'):
        """
        :param feat_type: Feature type to pool
        :param pool: Type of pooling: max or avg
        """
        super().__init__()
        assert pool in ['max', 'avg'], f'Unknown pooling: {pool}'
        assert feat_type == 0 or pool == 'avg', 'Max pooling on type > 0 features will break equivariance'
        self.feat_type = feat_type
        self.pool_type = pool
        # DGL pooling modules for DGLGraphWrapper
        self._dgl_pool = MaxPooling() if pool == 'max' else AvgPooling()

    def forward(self, features: Dict[str, Tensor], graph: SE3Graph, **kwargs) -> Tensor:
        feat = features[str(self.feat_type)]

        if isinstance(graph, DGLGraphWrapper):
            pooled = self._dgl_pool(graph._graph, feat)
        else:
            # PyTorch Geometric pooling for PyTorchGraph
            # PyG pooling expects [num_nodes, num_features], so flatten extra dims
            orig_shape = feat.shape
            feat_flat = feat.flatten()

            # Create batch assignment tensor from batch_num_nodes
            batch_num_nodes = graph.batch_num_nodes()
            batch = torch.repeat_interleave(
                torch.arange(batch_num_nodes.shape[0], device=feat.device),
                batch_num_nodes
            )

            if self.pool_type == 'max':
                pooled = global_max_pool(feat_flat, batch)
            else:
                pooled = global_mean_pool(feat_flat, batch)

            # Restore shape: [batch_size, *orig_shape[1:]]
            pooled = pooled.view(pooled.shape[0], *orig_shape[1:])

        return pooled.squeeze(dim=-1)
