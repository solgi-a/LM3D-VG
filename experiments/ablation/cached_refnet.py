"""
RefNet with the detection branch removed, for training on cached scene features.

Mirrors the second half of ``RefNet.forward`` (language branch -> proposal matching) and
nothing else. It is a separate model rather than a flag inside RefNet so that
models/refnet.py keeps its original content.

The detection branch is never *built*, not merely never called: constructing PointNet++ /
VoteNet / DETR would allocate the parameters, move them to GPU, and hand them to
``set_params_lr_dict``, which would put tensors that receive no gradient into the
optimizer. Skipping construction is where the speed and memory come from.

The tensors the detection branch would have written into ``data_dict`` come from
``cached_scenes.py :: CachedSceneDataset``; ``SCENE_CACHE_KEYS_REQUIRED`` there lists them
and where each one is consumed.
"""

import torch.nn as nn

from models.lang_module import LangModule
from models.match_module import MatchModule


class CachedRefNet(nn.Module):
    """Language + fusion only. Signature matches RefNet so callers can swap them."""

    def __init__(self, args, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=300, hidden_size=256, dataset_config=None):
        super().__init__()

        if no_reference:
            raise ValueError(
                "CachedRefNet trains the language and fusion branches; no_reference=True "
                "leaves it with nothing to train. Use RefNet for detection-only runs."
            )

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.no_reference = no_reference
        self.dataset_config = dataset_config
        self.args = args

        # --------- LANGUAGE ENCODING ---------
        # Same construction as RefNet.
        self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size)

        # --------- PROPOSAL MATCHING ---------
        # det_channel=288 matches RefNet; it is the width of the cached detr_features.
        self.match = MatchModule(args, num_proposals=num_proposal, det_channel=288)

    def forward(self, data_dict):
        """Same as RefNet.forward with the detection branch omitted.

        ``data_dict`` must already carry the cached detection outputs.
        """
        #######################################
        #                                     #
        #           LANGUAGE BRANCH           #
        #                                     #
        #######################################

        data_dict = self.lang(data_dict)

        #######################################
        #                                     #
        #          PROPOSAL MATCHING          #
        #                                     #
        #######################################

        data_dict = self.match(data_dict)

        return data_dict
