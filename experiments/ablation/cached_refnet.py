
import torch.nn as nn

from models.lang_module import LangModule
from models.match_module import MatchModule


class CachedRefNet(nn.Module):

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

        self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size)

        self.match = MatchModule(args, num_proposals=num_proposal, det_channel=288)

    def forward(self, data_dict):

        data_dict = self.lang(data_dict)


        data_dict = self.match(data_dict)

        return data_dict
