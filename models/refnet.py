import torch
import torch.nn as nn
import numpy as np
import sys
import os

from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.lang_module import LangModule
from models.match_module import MatchModule

from models.group_free.GF_detector import GroupFreeDetector

class RefNet(nn.Module):
    def __init__(self, args, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=300, hidden_size=256, dataset_config=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.no_reference = no_reference
        self.dataset_config = dataset_config

        self.args = args

        if args.detector == "GF":

            self.detector = GroupFreeDetector(num_class=num_class,
                                num_heading_bin=num_heading_bin,
                                num_size_cluster=num_size_cluster,
                                mean_size_arr=mean_size_arr,
                                input_feature_dim=input_feature_dim,
                                width=args.width,
                                bn_momentum=args.bn_momentum,
                                sync_bn=True if args.syncbn else False,
                                num_proposal=args.num_target,
                                sampling=args.sampling,
                                dropout=args.transformer_dropout,
                                activation=args.transformer_activation,
                                nhead=args.nhead,
                                num_decoder_layers=args.num_decoder_layers,
                                dim_feedforward=args.dim_feedforward,
                                self_position_embedding=args.self_position_embedding,
                                cross_position_embedding=args.cross_position_embedding,
                                size_cls_agnostic=True if args.size_cls_agnostic else False)

        else:

            # --------- PROPOSAL GENERATION ---------
            # Backbone point feature learning
            self.backbone_net = Pointnet2Backbone(args = args, input_feature_dim=self.input_feature_dim)

            # Hough voting
            self.vgen = VotingModule(self.vote_factor, 256)

            # Vote aggregation and object proposal
            config_transformer = None

            config_transformer = {
                'mask': 'no_mask',
                'weighted_input': True,
                'transformer_type': 'myAdd_20;deformable',
                'deformable_type': 'myAdd',
                'position_embedding': 'none',
                'input_dim': 0,
                'enc_layers': 0,
                'dec_layers': 2,
                'dim_feedforward': 2048,
                'hidden_dim': 288,
                'dropout': 0.1,
                'nheads': 8,
                'pre_norm': False
            }
            self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal,
                                        sampling, config_transformer=config_transformer, dataset_config=dataset_config)

        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            # self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size, det_channel=256*2)
            self.match = MatchModule(args, num_proposals=num_proposal, det_channel=288)  # bef 256

    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds,
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        if self.args.detector == "GF":
            
            data_dict = self.detector(data_dict)

        else:

            # --------- HOUGH VOTING ---------
            data_dict = self.backbone_net(data_dict)

            # --------- HOUGH VOTING ---------
            xyz = data_dict["fp2_xyz"]
            features = data_dict["fp2_features"]
            data_dict["seed_inds"] = data_dict["fp2_inds"]
            data_dict["seed_xyz"] = xyz
            data_dict["seed_features"] = features

            xyz, features = self.vgen(xyz, features)
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            data_dict["vote_xyz"] = xyz
            data_dict["vote_features"] = features

            # --------- PROPOSAL GENERATION ---------
            data_dict = self.proposal(xyz, features, data_dict)

        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang(data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            # config for bbox_embedding
            data_dict = self.match(data_dict)

        return data_dict
