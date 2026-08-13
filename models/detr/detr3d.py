# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .transformer3D import build_transformer, MLP


class DETR3D(nn.Module):
    """DETR module performing object detection; used as a backbone with encoding afterward."""
    def __init__(self, config_transformer, input_channels, class_output_shape, bbox_output_shape, aux_loss=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            input_channels: input channel of point cloud features
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        transformer_type = config_transformer.get('transformer_type', 'enc_dec')
        self.transformer_type = transformer_type
        if 'dec' in transformer_type:
            num_queries = config_transformer.num_queries
            self.num_queries = num_queries
        self.seed_attention = config_transformer.get('seed_attention', False)
        assert not self.seed_attention

        self.transformer = build_transformer(config_transformer)
        hidden_dim = self.transformer.d_model
        hidden_layer = config_transformer.dec_layers
        self.input_proj = nn.Linear(input_channels, hidden_dim)

        self.hidden_ffn = nn.Linear(hidden_dim * hidden_layer, hidden_dim)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, class_output_shape)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, bbox_output_shape, 2)

        if 'dec' in transformer_type:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        else:
            self.query_embed = None

        self.pos_embd_type = config_transformer.position_embedding
        self.mask_type = config_transformer.get('mask', 'detr_mask')

        self.weighted_input = config_transformer.get('weighted_input', False)
        if self.weighted_input:
            print('[INFO!] Use Weighted Input!')

        if self.pos_embd_type in ['self', 'none']:
            self.pos_embd = None
        self.aux_loss = aux_loss

    def forward(self, xyz, features, output, seed_xyz=None, seed_features=None, decode_vars=None):
        """Returns pred_logits (B x num_queries x num_classes+1), pred_boxes, and optional aux_outputs."""
        B, N, _ = xyz.shape
        _, _, C = features.shape

        if self.mask_type == 'detr_mask':
            mask = torch.zeros(B, N).bool().to(xyz.device)
            src_mask = None
        elif self.mask_type == 'no_mask':
            mask = None
            src_mask = None
        elif self.mask_type.split('_')[0] == 'near':
            near_kth = int(self.mask_type.split('_')[1])
            mask = None
            src_mask = torch.zeros(B, N, N).to(xyz.device) - 1e9
            A = xyz[:, None, :, :].repeat(1, N, 1, 1)
            B = xyz[:, :, None, :].repeat(1, 1, N, 1)
            dist = torch.sum((A - B).pow(2), dim=-1)

            dist_min, dist_pos = torch.topk(dist, k=near_kth, dim=1, largest=False, sorted=False)
            src_mask.scatter_(1, dist_pos, 0)
        else:
            raise NotImplementedError(self.mask_type)
        seed_embd = None
        if self.pos_embd_type == 'self':
            pos_embd = self.input_proj(features)
        elif self.pos_embd_type == 'none':
            pos_embd = None
        else:
            pos_embd = self.pos_embd(xyz)
            if seed_xyz is not None:
                seed_embd = self.pos_embd(seed_xyz)
        features = self.input_proj(features)
        query_embd_weight = self.query_embed.weight if self.query_embed is not None else None

        assert seed_xyz is None
        assert seed_features is None
        if self.weighted_input:
            value = self.transformer(features, mask, query_embd_weight, pos_embd, src_mask=src_mask, src_position=xyz)
        else:
            value = self.transformer(features, mask, query_embd_weight, pos_embd, src_mask=src_mask)

        # returns: dec_layer * B * Query * C
        if 'dec' in self.transformer_type or self.transformer_type.split(';')[-1] == 'deformable':
            hs = value[0]  # features_output
        elif self.transformer_type in ['enc']:
            hs = value
        else:
            raise NotImplementedError(self.transformer_type)
        detr_feat = hs.permute(1, 2, 0, 3).reshape(B, N, -1)
        detr_feat = nn.functional.relu(self.hidden_norm(self.hidden_ffn(detr_feat)))
        outputs_class = self.class_embed(detr_feat)
        outputs_coord = self.bbox_embed(detr_feat)
        if 'dec' in self.transformer_type or self.transformer_type.split(';')[-1] == 'deformable':
            output = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}  # final
            output['detr_features'] = detr_feat
            if self.aux_loss:
                output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

            if self.weighted_input or self.seed_attention:
                weighted_xyz = value[-1]
                output['transformer_weighted_xyz_all'] = weighted_xyz
                output['transformer_weighted_xyz'] = weighted_xyz[-1]
            else:
                raise NotImplementedError('must transformer weighted attn')
        else:
            raise NotImplementedError('only encoder not work')
        return output

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # workaround for torchscript, which rejects dicts with non-homogeneous values
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


if __name__ == "__main__":
    from easydict import EasyDict
    config_transformer = {
        'enc_layers': 6,
        'dec_layers': 6,
        'dim_feedforward': 2048,
        'hidden_dim': 288,
        'dropout': 0.1,
        'nheads': 8,
        'num_queries': 100,
        'pre_norm': False,
        'position_embedding': 'sine'
    }
    config_transformer = EasyDict(config_transformer)
    model = DETR3D(config_transformer, 128, 10, 20)
    xyz = torch.randn(4, 100, 3)
    features = torch.randn(4, 100, 128)
    out = model(xyz, features, {})
