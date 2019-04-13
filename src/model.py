import torch
import torch.nn as nn
import numpy as np

from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from pytorch_pretrained_bert.modeling import BertModel


class Head(nn.Module):
    """The MLP submodule"""
    def __init__(self, bert_hidden_size: int, linear_hidden_size: int, dist_embed_dim: int, token_dist_ratio: int, use_layers: list):
        super().__init__()
        self.bert_hidden_size = bert_hidden_size
        self.embed_dim = dist_embed_dim
        cat_contexts = 3
        self.use_layers = len(use_layers)
        self.span_extractor = SelfAttentiveSpanExtractor(bert_hidden_size * self.use_layers)
        # self.span_extractor = EndpointSpanExtractor(
        #     bert_hidden_size, "x,y,x*y"
        # )
        self.buckets = [-8, -4, -2, -1, 1, 2, 3, 4, 5, 8, 16, 32, 64]

        # self.fc_ABshare = nn.Sequential(
        #     nn.Linear(bert_hidden_size * self.use_layers, linear_hidden_size),
        #     nn.ReLU(),
        # )
        #
        # self.fc_Pshare = nn.Sequential(
        #     nn.Linear(bert_hidden_size * self.use_layers, linear_hidden_size),
        #     nn.ReLU(),
        # )


        self.fc_score = nn.Sequential(
            nn.BatchNorm1d(bert_hidden_size * cat_contexts * self.use_layers),
            nn.Dropout(0.1),
            nn.Linear(bert_hidden_size * cat_contexts * self.use_layers, linear_hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(linear_hidden_size),
            nn.Dropout(0.5),
            nn.Linear(linear_hidden_size, dist_embed_dim * token_dist_ratio)
        )
        self.fc_s = nn.Sequential(
            nn.BatchNorm1d(dist_embed_dim * (token_dist_ratio + 1)),
            nn.Dropout(0.5),
            nn.Linear(dist_embed_dim * (token_dist_ratio + 1), dist_embed_dim * (token_dist_ratio + 1)),
            nn.ReLU(),
        )
        self.fc_final = nn.Sequential(
            nn.Linear(dist_embed_dim * (token_dist_ratio + 1) * 2 + 2, 3)
        )
        self.dist_embed = nn.Embedding(len(self.buckets) + 1, embedding_dim=dist_embed_dim)

        for i, module in enumerate(self.fc_score):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                print("Initing batchnorm")
            elif isinstance(module, nn.Linear):
                if getattr(module, "weight_v", None) is not None:
                    nn.init.uniform_(module.weight_g, 0, 1)
                    nn.init.kaiming_normal_(module.weight_v)
                    print("Initing linear with weight normalization")
                    # assert model[i].weight_g is not None
                else:
                    nn.init.kaiming_normal_(module.weight)
                    print("Initing linear")
                nn.init.constant_(module.bias, 0)

        for i, module in enumerate(self.fc_final):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                print("Initing batchnorm")
            elif isinstance(module, nn.Linear):
                if getattr(module, "weight_v", None) is not None:
                    nn.init.uniform_(module.weight_g, 0, 1)
                    nn.init.kaiming_normal_(module.weight_v)
                    print("Initing linear with weight normalization")
                    # assert model[i].weight_g is not None
                else:
                    nn.init.kaiming_normal_(module.weight)
                    print("Initing linear")
                nn.init.constant_(module.bias, 0)

    def forward(self, bert_outputs, offsets, dists, in_urls):
        assert bert_outputs.size(2) == self.bert_hidden_size * self.use_layers
        spans_contexts = self.span_extractor(
            bert_outputs,
            offsets[:, :4].reshape(-1, 2, 2)
        )
        url_contexts = self.span_extractor(
             bert_outputs,
             offsets[:, 5:].reshape(-1, 1, 2)
         )[:, 0, :]
        dist_em = self.dist_embed(dists)
        dist_ema = dist_em[:, 0, :]
        dist_emb = dist_em[:, 1, :]
        #dist_em = dist_em.view(-1, 2 * self.embed_dim)
        # .reshape(offsets.size()[0], -1)
        # print(spans_contexts.shape)
        A_context = spans_contexts[:, 0, :]
        B_context = spans_contexts[:, 1, :]
        P_context = torch.gather(
                bert_outputs, 1,
                offsets[:, [4]].unsqueeze(2).expand(-1, -1, self.bert_hidden_size * self.use_layers)
            ).squeeze(1)

        PA = torch.cat([A_context * P_context, A_context, P_context], dim=1)
        PB = torch.cat([B_context * P_context, B_context, P_context], dim=1)

        score_PA = self.fc_score(PA)
        score_PB = self.fc_score(PB)

        # url_in_a = urls[:, [0]]
        # url_in_b = urls[:, [1]]

        SA = torch.cat([score_PA, dist_ema], dim=1)
        SB = torch.cat([score_PB, dist_emb], dim=1)

        score_A = self.fc_s(SA)
        score_B = self.fc_s(SB)

        a_in_url = in_urls[:, [0]]
        b_in_url = in_urls[:, [1]]

        score = torch.cat([score_A, score_B, a_in_url, b_in_url], dim=1)
        score = self.fc_final(score)

        return score

class GAPModel(nn.Module):
    """The main model."""
    def __init__(self, bert_model: str, device: torch.device, use_layer:list, linear_hidden_size=64,
                 dist_embed_dim=4, token_dist_ratio=4, bert_cache=None):
        super().__init__()
        self.device = device
        self.use_layer = use_layer

        self.bert_cache = bert_cache
        if bert_model in ("bert-base-uncased", "bert-base-cased"):
            self.bert_hidden_size = 768
        elif bert_model in ("bert-large-uncased", "bert-large-cased"):
            self.bert_hidden_size = 1024
        else:
            raise ValueError("Unsupported BERT model.")

        # self.bert = BertModel.from_pretrained(bert_model).to(device)
        self.bert = BertModel.from_pretrained(
            "/home/gy/.pytorch_pretrained_bert/214d4777e8e3eb234563136cd3a49f6bc34131de836848454373fa43f10adc5e.abfbb80ee795a608acbf35c7bf2d2d58574df3887cdd94b355fc67e03fddba05").to(
            device)
        # self.bert = BertModel.from_pretrained(bert_model).to(device)

        self.head = Head(self.bert_hidden_size, linear_hidden_size=linear_hidden_size, dist_embed_dim=dist_embed_dim, token_dist_ratio=token_dist_ratio, use_layers=use_layer).to(device)

    def forward(self, token_tensor, offsets, dists, in_urls):
        token_tensor = token_tensor.to(self.device)

        batch_size, token_len = token_tensor.shape

        bert_outputs = np.zeros((batch_size, 490, len(self.use_layer) * 1024), dtype=np.float32)
        miss_idx = []

        for idx, t in enumerate(token_tensor):
            key_numpy = np.zeros(490)
            key_numpy[:token_len] = t.cpu().numpy()
            key = tuple(key_numpy)
            if key not in self.bert_cache:
                miss_idx.append(idx)
            else:
                bert_outputs[idx, :, :] = self.bert_cache[key]

        if miss_idx:
            miss_token = torch.LongTensor(len(miss_idx), token_len).zero_().to(self.device)
            for idx, m_idx in enumerate(miss_idx):
                miss_token[idx, :] = token_tensor[m_idx, :]

            miss_bert_outputs, _ =  self.bert(
                miss_token, attention_mask=(miss_token > 0).long(),
                token_type_ids=None, output_all_encoded_layers=True)

            outputs = []
            for i in self.use_layer:
                outputs.append(miss_bert_outputs[i])
            miss_bert_outputs = torch.cat(outputs, dim=-1)

            for idx, m_idx in enumerate(miss_idx):
                bert_output = np.zeros((1, 490, len(self.use_layer) * 1024), dtype=np.float32)
                bert_output[:, :token_len, :] = miss_bert_outputs[[idx], :, :].cpu()
                bert_outputs[[m_idx], :, :] = bert_output
                key_numpy = np.zeros(490)
                key_numpy[:token_len] = miss_token[idx, :].cpu().numpy()
                key = tuple(key_numpy)
                self.bert_cache[key] = bert_output

        bert_outputs = torch.from_numpy(bert_outputs).to(self.device)

        # url_tensor = url_tensor.to(self.device)
        # url_outputs, _ = self.bert(
        #     url_tensor, attention_mask=(url_tensor > 0).long(),
        #     token_type_ids=None, output_all_encoded_layers=True)
        # url_outputs = torch.cat(url_outputs[self.use_layer:], dim=-1)

        head_outputs = self.head(bert_outputs, offsets.to(self.device), dists.to(self.device), in_urls.to(self.device))
        return head_outputs
