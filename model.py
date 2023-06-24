""" Componets of the model, based on MMdynamics
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from math import sqrt



            


class Lcoal_Confidence_Attention(nn.Module):
    def __init__(self, feature_dim, seq_len):
        super(Lcoal_Confidence_Attention, self).__init__()
        self.query_layer = nn.Linear(feature_dim, feature_dim)
        self.key_layer = nn.Linear(feature_dim, feature_dim)
        self.value_layer = nn.Linear(feature_dim, feature_dim)
        self.confidence = nn.Linear(feature_dim, seq_len)
        self._norm_fact = 1 / sqrt(feature_dim)

    def forward(self, query, key, value):
        "q k v : batch_size*seq_len*feature_dim "

        batch_size, seq_len, feature_dim = query.shape

        query = torch.relu(self.query_layer(query))
        key = torch.relu(self.key_layer(key))
        value = torch.relu(self.value_layer(value))

        # apply attention mechanism
        output = torch.bmm(query, key.transpose(1, 2)) * self._norm_fact
        attn_scores = torch.softmax(output, dim=-1)

        # confidence evaluation
        confidence_scores = self.confidence(key)
        confidence_scores = torch.sigmoid(confidence_scores)

        LCA_Loss = torch.mean(F.mse_loss(attn_scores, confidence_scores))


        attn_values = torch.matmul(attn_scores, value)
        attn_values = attn_values.view(batch_size, seq_len*feature_dim)


        return LCA_Loss,  attn_values

class Global_Confidence_Attention(nn.Module):
    def __init__(self, feature_dim, seq_len, modality_num):
        super(Global_Confidence_Attention, self).__init__()
        self.modality_num = modality_num
        self.ProjectLayerList = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for m in range(self.modality_num)])
        self.confidence = nn.Linear(feature_dim, seq_len)
        self._norm_fact = 1 / sqrt(feature_dim)

    def forward(self, modality):
        " modality1: modality_num*batch_size*seq_len*feature_dim "

        modality_num, batch_size, seq_len, feature_dim = modality.shape

        modality_proj = []
        for i in range(modality_num):
            modality_proj.append(torch.relu(self.ProjectLayerList[i](modality[i])))

        modality_proj_cat = torch.stack(modality_proj)

        attn_scores = [] # modality_num*cross_modality_att_num*batch_size*seq_len*seq_len
        attn_values = [] # modality_num*cross_modality_att_num*batch_size*seq_len*feature_dim

        # global attention mechanism
        for i in range(modality_num):
            temp_qk = []
            temp_v = []
            for j in range(modality_num):
                temp_qk.append(torch.relu(torch.bmm(modality_proj[i], modality_proj[j].transpose(1, 2)) * self._norm_fact))
                temp_v.append(modality_proj[j])
            output_qk = torch.stack(temp_qk)   # cross_modality_att_num*batch_size*seq_len*seq_len
            output_v = torch.stack(temp_v)
            output_qk = torch.softmax(output_qk, dim=-1)
            values = torch.matmul(output_qk, output_v)
            attn_scores.append(output_qk)
            attn_values.append(values)

        # confidence evaluation
        attn_scores_ = attn_scores[0]
        for i in range(1,self.modality_num):
            attn_scores_ = attn_scores_ + attn_scores[i]

        attn_scores = torch.stack(attn_scores)
        attn_values = torch.stack(attn_values)

        confidence_scores = self.confidence(modality_proj_cat)
        confidence_scores = torch.sigmoid(confidence_scores)

        GCA_Loss = torch.mean(F.mse_loss(attn_scores_, confidence_scores))

        attn_values = torch.matmul(attn_scores, attn_values)
        attn_values = attn_values.view(batch_size, modality_num*modality_num*seq_len*feature_dim)

        return GCA_Loss,  attn_values





