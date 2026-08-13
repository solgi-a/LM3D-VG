import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from copy import deepcopy

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.transformer.attention import MultiHeadAttention

class LangModule(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, use_bidir=False,
                 emb_size=300, hidden_size=256, final_size=128):
        super().__init__()

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.use_bidir
        )
        lang_size = hidden_size * 2 if self.use_bidir else hidden_size

        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_size, num_text_classes),
                nn.Dropout()
            )

        self.ML = nn.Sequential(nn.Linear(hidden_size, final_size), nn.ReLU(),nn.Dropout(p=.1),nn.LayerNorm(final_size))
        self.mhatt = MultiHeadAttention(d_model=final_size, d_k=16, d_v=16, h=4, dropout=.1, identity_map_reordering=False,
                                        attention_module=None,
                                        attention_module_kwargs=None)

    def _lang_model_forward(self, word_embs, lang_len):

        lang_feat = pack_padded_sequence(word_embs, lang_len.cpu(), batch_first=True, enforce_sorted=False)  # Note: For high-version cuda: .cpu()
        #lang_feat = pack_padded_sequence(word_embs, lang_len, batch_first=True, enforce_sorted=False)

        out, lang_last = self.gru(lang_feat)

        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        if self.use_bidir:
            cap_emb = (cap_emb[:, :, :int(cap_emb.shape[2] / 2)] + cap_emb[:, :, int(cap_emb.shape[2] / 2):]) / 2

        b_s, seq_len = cap_emb.shape[:2]
        mask_queries = torch.ones((b_s, seq_len), dtype=torch.int)
        for i in range(b_s):
            mask_queries[i, cap_len[i]:] = 0
        attention_mask = (mask_queries == 0).unsqueeze(1).unsqueeze(1).cuda()  # (b_s, 1, 1, seq_len)
                
        lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1)  # batch_size, hidden_size * num_dir

        return cap_emb, lang_last, attention_mask

    def forward(self, data_dict):
        word_embs = data_dict["lang_feat_list"]  # B * 32 * MAX_DES_LEN * LEN(300)
        lang_len = data_dict["lang_len_list"]
        batch_size, len_nun_max, max_des_len = word_embs.shape[:3]

        word_embs = word_embs.reshape(batch_size * len_nun_max, max_des_len, -1)
        lang_len = lang_len.reshape(batch_size * len_nun_max)
        first_obj = data_dict["first_obj_list"].reshape(batch_size * len_nun_max)


        target = data_dict['target']  # B * 32 * MAX_DES_LEN * LEN(300)
        tgt_len = data_dict["tgt_len"]
        adjectives = data_dict['adjectives']  # B * 32 * MAX_DES_LEN * LEN(300)
        adj_len = data_dict["adj_len"]
        neighbors = data_dict['neighbors']  # B * 32 * MAX_DES_LEN * LEN(300)
        ngh_len = data_dict["ngh_len"]

        target = target.reshape(batch_size * len_nun_max, target.shape[2], -1)
        tgt_len = tgt_len.reshape(batch_size * len_nun_max)
        adjectives = adjectives.reshape(batch_size * len_nun_max, adjectives.shape[2], -1)
        adj_len = adj_len.reshape(batch_size * len_nun_max)
        neighbors = neighbors.reshape(batch_size * len_nun_max, neighbors.shape[2], -1)
        ngh_len = ngh_len.reshape(batch_size * len_nun_max)

        random_numer = random.random()

        if data_dict["istrain"][0] == 1:
            for i in range(word_embs.shape[0]):
                
                if random_numer < 0.5:

                    main_object_name = word_embs[i, first_obj[i]].unsqueeze(0)
                    
                    comparison = (target[i] == main_object_name)
                    matches = comparison.all(dim=1)
                    target[i][matches] = data_dict["unk"][0]
                    
                    word_embs[i, first_obj[i]] = data_dict["unk"][0]

                sen_len = lang_len[i]

                random_numbers = torch.randperm(sen_len)[:int(sen_len/5)]
                masked_words = word_embs[i, random_numbers]

                for mask_iter in range(len(masked_words)):

                    iter_masked_word = masked_words[mask_iter].unsqueeze(0)

                    comparison = (target[i] == iter_masked_word)
                    matches = comparison.all(dim=1)
                    target[i][matches] = data_dict["unk"][0]

                    comparison = (adjectives[i] == iter_masked_word)
                    matches = comparison.all(dim=1)
                    adjectives[i][matches] = data_dict["unk"][0]

                    comparison = (neighbors[i] == iter_masked_word)
                    matches = comparison.all(dim=1)
                    neighbors[i][matches] = data_dict["unk"][0]

                word_embs[i, random_numbers] = data_dict["unk"][0]

                """ for j in range(int(sen_len/5)):
                    
                    num = random.randint(0, sen_len-1)
                    masked_word = word_embs[i, num].unsqueeze(0)

                    comparison = (adjectives[i] == masked_word)
                    matches = comparison.all(dim=1)
                    adjectives[i][matches] = data_dict["unk"][0]

                    comparison = (neighbors[i] == masked_word)
                    matches = comparison.all(dim=1)
                    neighbors[i][matches] = data_dict["unk"][0]

                    #num_adj_match = (adjectives[i] == masked_word).sum(1)/300 == 1
                    #adjectives[i][num_adj_match] = data_dict["unk"][0]

                    #num_ngh_match = (neighbors[i] == masked_word).sum(1)/300 == 1
                    #neighbors[i][num_ngh_match] = data_dict["unk"][0]
                    word_embs[i, num] = data_dict["unk"][0] """

        """ elif data_dict["istrain"][0] == 1:
            for i in range(word_embs.shape[0]):

                sen_len = lang_len[i]

                for j in range(int(sen_len/5)):

                    num = random.randint(0, sen_len-1)

                    masked_word = word_embs[i, num].unsqueeze(0)

                    comparison = (adjectives[i] == masked_word)
                    matches = comparison.all(dim=1)
                    adjectives[i][matches] = data_dict["unk"][0]

                    comparison = (neighbors[i] == masked_word)
                    matches = comparison.all(dim=1)
                    neighbors[i][matches] = data_dict["unk"][0]

                    #num_adj_match = (adjectives[i] == masked_word).sum(1)/300 == 1
                    #adjectives[i][num_adj_match] = data_dict["unk"][0]

                    #num_ngh_match = (neighbors[i] == masked_word).sum(1)/300 == 1
                    #neighbors[i][num_ngh_match] = data_dict["unk"][0]
                    
                    word_embs[i, num] = data_dict["unk"][0] """


        main_lang_len = data_dict["main_lang_len_list"]
        main_lang_len = main_lang_len.reshape(batch_size * len_nun_max)

        if data_dict["istrain"][0] == 1 and random.random() < 0.5:
            for i in range(word_embs.shape[0]):
                new_word_emb = copy.deepcopy(word_embs[i])
                new_len = lang_len[i] - main_lang_len[i]
                new_word_emb[:new_len] = word_embs[i, main_lang_len[i]:lang_len[i]]
                new_word_emb[new_len:lang_len[i]] = word_embs[i, :main_lang_len[i]]
                word_embs[i] = new_word_emb


        cap_emb, lang_last, attention_mask = self._lang_model_forward(word_embs, lang_len)
        data_dict["attention_mask"] = attention_mask

        lang_fea = self.ML(cap_emb)
        lang_fea = self.mhatt(lang_fea, lang_fea, lang_fea, attention_mask)

        data_dict["lang_fea"] = lang_fea
        data_dict["lang_emb"] = lang_last  # B, hidden_size


        tgt_cap_emb, _, tgt_attention_mask = self._lang_model_forward(target, tgt_len)
        tgt_lang_fea = self.ML(tgt_cap_emb)
        tgt_lang_fea = self.mhatt(tgt_lang_fea, tgt_lang_fea, tgt_lang_fea, tgt_attention_mask)

        adj_cap_emb, _, adj_attention_mask = self._lang_model_forward(adjectives, adj_len)
        adj_lang_fea = self.ML(adj_cap_emb)
        adj_lang_fea = self.mhatt(adj_lang_fea, adj_lang_fea, adj_lang_fea, adj_attention_mask)

        ngh_cap_emb, _, ngh_attention_mask = self._lang_model_forward(neighbors, ngh_len)
        ngh_lang_fea = self.ML(ngh_cap_emb)
        ngh_lang_fea = self.mhatt(ngh_lang_fea, ngh_lang_fea, ngh_lang_fea, ngh_attention_mask)

        data_dict["tgt_attention_mask"] = tgt_attention_mask
        data_dict["adj_attention_mask"] = adj_attention_mask
        data_dict["ngh_attention_mask"] = ngh_attention_mask

        data_dict["tgt_lang_fea"] = tgt_lang_fea
        data_dict["adj_lang_fea"] = adj_lang_fea
        data_dict["ngh_lang_fea"] = ngh_lang_fea

        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])

        return data_dict

