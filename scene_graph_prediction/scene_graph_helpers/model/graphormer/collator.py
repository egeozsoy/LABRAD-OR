# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import pyximport
import numpy as np

pyximport.install(setup_args={'include_dirs': np.get_include()})
from scene_graph_prediction.scene_graph_helpers.model.graphormer import algos


def convert_to_single_emb(x, offset=32):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
                     torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
    ] = convert_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()

    return item


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


class Batch():
    def __init__(self, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, x, edge_input, meta):
        super(Batch, self).__init__()
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x = x
        self.attn_bias, self.attn_edge_type, self.spatial_pos = attn_bias, attn_edge_type, spatial_pos
        self.edge_input = edge_input
        self.meta = meta

    def to(self, device):
        self.in_degree, self.out_degree = self.in_degree.to(
            device), self.out_degree.to(device)
        self.x = self.x.to(device)
        self.attn_bias, self.attn_edge_type, self.spatial_pos = self.attn_bias.to(
            device), self.attn_edge_type.to(device), self.spatial_pos.to(device)
        self.edge_input = self.edge_input.to(device)
        for k, v in self.meta.items():
            if torch.is_tensor(v):
                self.meta[k] = v.to(device)
        return self

    def __len__(self):
        return self.in_degree.size(0)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = torch.arange(start, stop, step, dtype=torch.long)
        else:
            indices = torch.tensor([index], dtype=torch.long)

        in_degree = self.in_degree[indices]
        out_degree = self.out_degree[indices]
        x = self.x[indices]
        attn_bias = self.attn_bias[indices]
        attn_edge_type = self.attn_edge_type[indices]
        spatial_pos = self.spatial_pos[indices]
        edge_input = self.edge_input[indices]

        meta = {}
        for k, v in self.meta.items():
            if torch.is_tensor(v):
                meta[k] = v[indices]
            else:
                meta[k] = v

        return Batch(attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, x, edge_input, meta)


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    meta = {}
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :]) for item in items]
    attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                             for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                           for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                            for i in out_degrees])
    return Batch(
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=out_degree,
        x=x,
        edge_input=edge_input,
        meta=meta
    )
