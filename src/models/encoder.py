import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neural import MultiHeadedAttention, PositionwiseFeedForward


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


# Modified Global Attention
class RelativeGlobalAttention(nn.Module):
    def __init__(self, dropout, d_model, num_heads, max_len=1024):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len))
                .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )

        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)

        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)

        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


# End Modification


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtLayer(nn.Module):
    def __init__(self, transformer, bert_used, d_model, d_ff, heads, dropout, num_inter_layers=0, doc_weight = 0.8, extra_attention = False):
        super(ExtLayer, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        if extra_attention:
            self.extra_attention = True
            self.global_attention = RelativeGlobalAttention(dropout, d_model, heads)
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer = transformer
        self.bert_used = bert_used
        self.doc_weight = doc_weight
        if bert_used == 'distilbert':
            self.transformer_inter = transformer.layer
        elif bert_used == 'albert':
            self.transformer_inter = transformer.albert_layer_groups
        else:
            self.transformer_inter = nn.ModuleList(
                [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
                 for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask, head_mask=None):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        if self.bert_used == 'distilbert':
            head_mask = [None] * self.transformer.n_layers if head_mask is None else head_mask
            x, = self.transformer(x, mask, head_mask) # Doc level representation
            # for i in range(self.transformer.n_layers):
            #     x, = self.transformer_inter[i](x, mask)  # all_sents * max_tokens * dim (Modified Masking for Pytorch above v1.2)
        elif self.bert_used == 'albert':
            head_mask = [None] * self.transformer.config.num_hidden_layers if head_mask is None else head_mask
            for i in range(self.transformer.config.num_hidden_layers):
                layers_per_group = int(self.transformer.config.num_hidden_layers / self.transformer.config.num_hidden_groups)
                group_idx = int(i / (self.transformer.config.num_hidden_layers / self.transformer.config.num_hidden_groups))
                # Attention mask below follows albert model way of extension
                extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                layer_group_output = self.transformer_inter[group_idx](x, extended_attention_mask, head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group])
                x = layer_group_output[0]

        x = (1-self.doc_weight)*x + (self.doc_weight)*top_vecs
        if self.extra_attention == True:
            x = self.global_attention(x)
        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, 1 - mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores

# class SentEncoder(nn.Module):
#     def __init__(self, bert, ext_layer, args, device):
#         super(SentEncoder, self).__init__()
#         self.args = args
#         self.device = device
#         self.bert = bert
#         self.ext_layer = ext_layer
#         # self.ext_layer = self.bert.model.transformer
#
#         self.to(device)
#
#     def forward(self, src, segs, clss, mask_src, mask_cls):
#         top_vec = self.bert(src, segs, mask_src)    # top_vec: Tensor(num_batch, token_length, hidden_size)
#         sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]   # sents_vec: Tensor(num_batch, num_sents, hidden_size)
#         sents_vec = sents_vec * mask_cls[:, :, None].float()    # sents_vec: Tensor(num_batch, num_sents, hidden_size)
#         sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)   # sent_scores: Tensor(num_batch, num_sents)
#         return sent_scores, mask_cls
#
#
# class DocEncoder(nn.Module):
#     def __init__(self, bert, ext_layer, args, device):
#         super(DocEncoder, self).__init__()
#         self.args = args
#         self.device = device
#         self.bert = bert
#         self.ext_layer = ext_layer
#
#         self.to(device)
#
#     def forward(self, sents_vec, segs, mask_sents):
#         doc_embedding = torch.zeros(1, 1, self.bert.model.config.hidden_size).to(self.device)   # doc_embedding: Tensor(1, 1, hidden_size)
#         doc_embedding[:, :, :sents_vec.size(1)] = sents_vec # doc_embedding: Tensor(1, 1, hidden_size) # Added sent scores
#         doc_score = self.ext_layer(doc_embedding, mask_sents).squeeze(-1)
#         return doc_score
#

