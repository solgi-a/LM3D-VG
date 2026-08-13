import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import PositionWiseFeedForward
from torch_geometric.nn import knn_graph, GCNConv
from experiments.ablation.ablation_config import ABLATION  # ABLATION
import random

class MatchModule(nn.Module):
    def __init__(self, args, num_proposals=256, hidden_size=128, det_channel=288, head=4, depth=2):
        super().__init__()
        self.use_dist_weight_matrix = True  ## False: initial 3DVG-Transformer

        self.num_proposals = num_proposals
        self.hidden_size = hidden_size
        self.depth = depth - 1
        self.args = args
        
        self.features_concat = nn.Sequential(
            nn.Conv1d(det_channel, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
        )

        self.match = nn.Sequential(
            nn.Conv1d(2*hidden_size, 2*hidden_size, 1),
            nn.BatchNorm1d(2*hidden_size),
            nn.PReLU(),

            nn.Conv1d(2*hidden_size, 2*hidden_size, 1),
            nn.BatchNorm1d(2*hidden_size),
            nn.PReLU(),

            nn.Conv1d(2*hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),

            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),

            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),

            nn.Conv1d(hidden_size, 1, 1)
        )

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))
        
        #self.self_attn = MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)

        self.cross_attn = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))  # k, q, v

        self.tgt_cross_attn = MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)  # k, q, v

        self.adj_cross_attn = MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)  # k, q, v

        self.ngh_cross_attn = MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)  # k, q, v
        
        self.gcn1 = GCNConv(hidden_size, hidden_size, improved=True, add_self_loops=True)
        
        self.gcn2 = GCNConv(hidden_size, hidden_size, improved=True, add_self_loops=True)

        #self.bbox_embedding = nn.Linear(12, 128)

        
    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.use_dist_weight_matrix:
            # Attention Weight
            objects_center = data_dict['center']
            N_K = objects_center.shape[1]
            center_A = objects_center[:, None, :, :].repeat(1, N_K, 1, 1)
            center_B = objects_center[:, :, None, :].repeat(1, 1, N_K, 1)
            dist = (center_A - center_B).pow(2)
            # print(dist.shape, '<< dist shape', flush=True)
            dist = torch.sqrt(torch.sum(dist, dim=-1))[:, None, :, :]
            dist_weights = 1 / (dist+1e-2)
            norm = torch.sum(dist_weights, dim=2, keepdim=True)
            dist_weights = dist_weights / norm
            zeros = torch.zeros_like(dist_weights)

            dist_weights = torch.cat([dist_weights, -dist, zeros, zeros], dim=1).detach()
            attention_matrix_way = 'add'
        else:
            dist_weights = None
            attention_matrix_way = 'mul'


        # object size embedding
        # print(data_dict.keys())
        if self.args.detector == "VN":
            features = data_dict['detr_features']
            features = features.permute(0, 2, 1)

        else:
            features = data_dict['features']

        features = self.features_concat(features)
        features = features.permute(0, 2, 1)

        batch_size, num_proposal = features.shape[:2]

        if data_dict['objectness_scores'].shape[2] == 2:
            objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)  # batch_size, num_proposals, 1
        else:
            objectness_masks = (data_dict['objectness_scores']>0).float()

        #features = self.mhatt(features, features, features, proposal_masks)
        features = self.self_attn[0](features, features, features, attention_weights=dist_weights, way=attention_matrix_way)

        len_nun_max = data_dict["lang_feat_list"].shape[1]

        #####################################################################################################################
        obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
        obj_lens = torch.zeros(batch_size, dtype=torch.int).cuda()

        for i in range(batch_size):
            obj_mask = torch.where(obj_masks[i, :] == True)[0]
            obj_len = obj_mask.shape[0]
            obj_lens[i] = obj_len

        k_neighbors = 5 if obj_lens.min() >= 5 else obj_lens.min()

        graph_obj_masks = (objectness_masks.bool().squeeze(2)).reshape(batch_size*num_proposal)
        graph_obj_center = data_dict['center'].reshape(batch_size*num_proposal, -1)

        graph_obj_features = features.clone()
        graph_obj_features = graph_obj_features.reshape(batch_size*num_proposal, -1)

        masked_pos = graph_obj_center[graph_obj_masks]
        masked_features = graph_obj_features[graph_obj_masks]
        
        # Update batch tensor according to the graph_obj_masks
        masked_batch = torch.arange(batch_size, device=graph_obj_masks.device).repeat_interleave(num_proposal)[graph_obj_masks]

        # Construct the k-NN graph based on the masked spatial positions
        edge_index = knn_graph(masked_pos, k=k_neighbors, batch=masked_batch)

        output_features = self.gcn1(masked_features, edge_index)
        output_features = self.gcn2(output_features, edge_index)

        # restore the output to the original shape, filling in zeros for masked-out proposals
        #full_output_features = torch.zeros(batch_size * num_proposal, self.hidden_size).cuda()
        full_output_features = graph_obj_features.clone()
        full_output_features[graph_obj_masks] = output_features
        grpah_features = full_output_features.view(batch_size, num_proposal, self.hidden_size)

        grpah_features = self.self_attn[1](grpah_features, grpah_features, grpah_features, attention_weights=dist_weights, way=attention_matrix_way)

        #--------------------------------------------------------------------------------------------------------------------

        #data_dict["random"] = random.random()
        random_numer = random.random()

        # copy paste
        grpah_features_output = grpah_features.clone()
        if data_dict["istrain"][0] == 1 and random_numer < 0.5 and not ABLATION.DISABLE_COPY_PASTE:  # ABLATION
            obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
            obj_lens = torch.zeros(batch_size, dtype=torch.int).cuda()
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == True)[0]
                obj_len = obj_mask.shape[0]
                obj_lens[i] = obj_len

            obj_masks_reshape = obj_masks.reshape(batch_size*num_proposal)
            obj_features = grpah_features.reshape(batch_size*num_proposal, -1)
            obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
            total_len = obj_mask.shape[0]
            obj_features = obj_features[obj_mask, :].repeat(2,1)  # total_len, hidden_size
            j = 0
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == False)[0]
                obj_len = obj_mask.shape[0]
                j += obj_lens[i]
                if obj_len < total_len - obj_lens[i]:
                    grpah_features_output[i, obj_mask, :] = obj_features[j:j + obj_len, :]
                else:
                    grpah_features_output[i, obj_mask[:total_len - obj_lens[i]], :] = obj_features[j:j + total_len - obj_lens[i], :]

        feature2 = grpah_features_output[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, num_proposal, -1)

        #####################################################################################################################

        data_dict["random"] = random.random()

        # copy paste
        feature0 = features.clone()
        if data_dict["istrain"][0] == 1 and data_dict["random"] < 0.5 and not ABLATION.DISABLE_COPY_PASTE:  # ABLATION
            obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
            obj_lens = torch.zeros(batch_size, dtype=torch.int).cuda()
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == True)[0]
                obj_len = obj_mask.shape[0]
                obj_lens[i] = obj_len

            obj_masks_reshape = obj_masks.reshape(batch_size*num_proposal)
            obj_features = features.reshape(batch_size*num_proposal, -1)
            obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
            total_len = obj_mask.shape[0]
            obj_features = obj_features[obj_mask, :].repeat(2,1)  # total_len, hidden_size
            j = 0
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == False)[0]
                obj_len = obj_mask.shape[0]
                j += obj_lens[i]
                if obj_len < total_len - obj_lens[i]:
                    feature0[i, obj_mask, :] = obj_features[j:j + obj_len, :]
                else:
                    feature0[i, obj_mask[:total_len - obj_lens[i]], :] = obj_features[j:j + total_len - obj_lens[i], :]

        feature1 = feature0[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, num_proposal, -1)
        if dist_weights is not None:
            dist_weights = dist_weights[:, None, :, :, :].repeat(1, len_nun_max, 1, 1, 1).reshape(batch_size*len_nun_max, dist_weights.shape[1], num_proposal, num_proposal)

        #--------------------------------------------------------------------------------------------------------------------

        lang_fea = data_dict["lang_fea"]
        tgt_lang_fea = data_dict["tgt_lang_fea"]
        adj_lang_fea = data_dict["adj_lang_fea"]
        ngh_lang_fea = data_dict["ngh_lang_fea"] 
        
        feature1 = self.cross_attn[0](feature1, lang_fea, lang_fea, data_dict["attention_mask"])

        #for _ in range(self.depth):
            #feature1 = self.self_attn[_+1](feature1, feature1, feature1, attention_weights=dist_weights, way=attention_matrix_way)
            #feature1 = self.cross_attn[_+1](feature1, lang_fea, lang_fea, data_dict["attention_mask"])

        feature1 = self.tgt_cross_attn(feature1, tgt_lang_fea, tgt_lang_fea, data_dict["tgt_attention_mask"])
        feature1 = self.adj_cross_attn(feature1, adj_lang_fea, adj_lang_fea, data_dict["adj_attention_mask"])

        #--------------------------------------------------------------------------------------------------------------------

        feature2 = self.cross_attn[1](feature2, lang_fea, lang_fea, data_dict["attention_mask"])
        feature2 = self.ngh_cross_attn(feature2, ngh_lang_fea, ngh_lang_fea, data_dict["ngh_attention_mask"])
        
        # match
        feature_agg = torch.cat((feature1,feature2),dim=-1)
        feature_agg = feature_agg.permute(0, 2, 1).contiguous()

        confidence = self.match(feature_agg).squeeze(1)  # batch_size, num_proposals
        data_dict["cluster_ref"] = confidence

        return data_dict
