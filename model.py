import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel

class TextCNN(nn.Module):
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embedding_dim=300,
                 output_channel=100,
                 kernel_wins=None,
                 num_classes= 2):
        super(TextCNN, self).__init__()

        if kernel_wins is None:
            kernel_wins = [2, 3, 4, 5]

        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embedding_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=-1,
                                          max_norm=5.0)

        self.convs = nn.ModuleList([nn.Conv1d(self.embed_dim, output_channel, kernel_size) for kernel_size in kernel_wins])

        self.dropout = nn.Dropout(0.8)
        # fc
        self.fc = nn.Linear(len(kernel_wins)*output_channel, num_classes)

    def forward(self, input_ids):

        # [batch_size, sequence_length, embedding_size]
        embedding_x = self.embedding(input_ids).float()

        # [batch, embedding_size, sequence_length]
        reshaped_x = embedding_x.permute(0, 2, 1)

        # [batch_size, num_filters[i], L_out]
        con_x_list = [F.relu_(conv(reshaped_x)) for conv in self.convs]

        # (batch_size, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(con_x, kernel_size=con_x.shape[2])
                       for con_x in con_x_list]

        fc_x = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        fc_x = self.dropout(fc_x)

        output = self.fc(fc_x)

        return output

class BaseCNN(nn.Module):
    def __init__(self,
                 embedding_dim=300,
                 num_classes= 2):
        super(BaseCNN, self).__init__()

        self.convs = nn.Sequential(
                nn.Conv1d()
            )


    def forward(self, input_ids):

        # [batch_size, sequence_length, embedding_size]
        embedding_x = self.embedding(input_ids).float()

        # [batch, embedding_size, sequence_length]
        reshaped_x = embedding_x.permute(0, 2, 1)

        # [batch_size, num_filters[i], L_out]
        con_x_list = [F.relu_(conv(reshaped_x)) for conv in self.convs]

        # (batch_size, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(con_x, kernel_size=con_x.shape[2])
                       for con_x in con_x_list]

        fc_x = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        fc_x = self.dropout(fc_x)

        output = self.fc(fc_x)

        return output


class LSTM(nn.Module):

    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embedding_dim=300,
                 hidden_dim=128,
                 layer_dim=3,
                 num_classes=2):
        super(LSTM, self).__init__()

        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embedding_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=-1,
                                          max_norm=5.0)

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, input_lens):

        embedding_x = self.embedding(input_ids).float()

        pack_input = pack_padded_sequence(input=embedding_x, lengths=input_lens, batch_first=True)

        r_out, (h_n, h_c) = self.lstm(pack_input, None)

        output = self.fc(h_n[-1, :, :])

        return output


class Bi_LSTM(nn.Module):
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embedding_dim=300,
                 hidden_dim=128,
                 layer_dim=3,
                 num_classes=2):
        super(Bi_LSTM, self).__init__()

        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embedding_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=-1,
                                          max_norm=5.0)

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True,
                            bias=True, dropout=0.5)

        # self.drop = nn.Dropout(p=0.2)
        # self.fc = nn.Linear(hidden_dim*2, num_classes)
        # self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, num_classes)
        )

    def forward(self, input_ids, input_lens):

        embedding_x = self.embedding(input_ids).float()

        pack_input = pack_padded_sequence(input=embedding_x, lengths=input_lens, batch_first=True)

        r_out, (h_n, h_c) = self.lstm(pack_input, None)

        output = self.fc(torch.cat((h_n[-1, :, :], h_n[-2, :, :]), 1))
        return output

class RCNN(nn.Module):
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embedding_dim=300,
                 hidden_dim=128,
                 layer_dim=3,
                 num_classes=2):
        super(RCNN, self).__init__()

        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embedding_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=-1,
                                          max_norm=5.0)

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True,
                            bias=True, dropout=0.5)

        # self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_dim * 2 + embedding_dim, num_classes)


    def forward(self, input_ids, input_lens):

        embedding_x = self.embedding(input_ids).float()

        pack_input = embedding_x[:, 0:input_lens[0], :]

        r_out, (h_n, h_c) = self.lstm(pack_input, None)

        left_placehoder = Variable(torch.zeros(1, 1, self.hidden_dim))
        left_lstm = torch.cat((left_placehoder, r_out[:, 0:-1, 0:int(self.hidden_dim)]), dim=1)

        right_placehoder = Variable(torch.zeros(1, 1, self.hidden_dim))
        right_lstm = torch.cat((r_out[:, 1:, int(self.hidden_dim):], right_placehoder), dim=1)

        # 前 + embedding + 后
        lstm_cat = torch.cat((left_lstm, pack_input, right_lstm), 2)
        y1 = torch.tanh(lstm_cat.permute(0, 2, 1))
        y2 = F.max_pool1d(y1, y1.size()[2])
        y3 = self.fc(torch.squeeze(y2, dim=2))

        return y3


class My_BertModel(nn.Module):
    def __init__(self, MODEL_PATH, freeze_bert=True, use_all_layer=False):
        super(My_BertModel, self).__init__()
        self.use_all_layer = use_all_layer
        self.model = BertModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, output_hidden_states=True,
                                               output_attentions=True, return_dict=True)
        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False

        # share layer
        self.softmax_all_layer = nn.Softmax(-1)
        self.nn_dense = nn.Linear(self.model.config.hidden_size, 1)
        # use a truncated_normalizer to initialize the α.
        self.truncated_normal_(self.nn_dense.weight)
        self.act = nn.ReLU()
        self.pooler = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.pooler_activation = nn.Tanh()

        # is_humour
        self.subtask_1a = nn.Sequential(
            nn.Dropout(p=0.9),
            nn.Linear(self.model.config.hidden_size, 2)
        )

    # this function is adapted form https://zhuanlan.zhihu.com/p/83609874
    def truncated_normal_(self, tensor, mean=0, std=0.02):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # calculate α_i
        layer_logits = []
        for layer in outputs.hidden_states[1:]:
            out = self.nn_dense(layer)
            layer_logits.append(self.act(out))

        # sum up layers by weighting
        layer_logits = torch.cat(layer_logits, axis=2)
        layer_dist = self.softmax_all_layer(layer_logits)
        seq_out = torch.cat([torch.unsqueeze(x, axis=2) for x in outputs.hidden_states[1:]], axis=2)
        all_layer_output = torch.matmul(torch.unsqueeze(layer_dist, axis=2), seq_out)
        all_layer_output = torch.squeeze(all_layer_output, axis=2)
        # take the [CLS] token output
        all_layer_output = self.pooler_activation(
            self.pooler(all_layer_output[:, 0])) if self.pooler is not None else None

        if not self.use_all_layer:
            # use [CLS] tokken output for the last layer encoder
            pooled_output = outputs.pooler_output
        else:
            pooled_output = all_layer_output

        output = self.subtask_1a(pooled_output)

        return output
