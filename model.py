import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embedding_dim=300,
                 output_channel=100,
                 kernel_wins=None,
                 num_classes= 1):
        super(TextCNN, self).__init__()

        if kernel_wins is None:
            kernel_wins = [3, 4, 5]

        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embedding_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)

        self.convs = nn.ModuleList([nn.Conv2d(self.embed_dim, output_channel, kernel_size) for kernel_size in kernel_wins])

        self.dropout = nn.Dropout(0.5)
        # fc
        self.fc = nn.Linear(len(kernel_wins)*output_channel, num_classes)

    def forward(self, X):

        # [batch_size, sequence_length, embedding_size]
        embedding_x = self.W(X)

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
