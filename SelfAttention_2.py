# -*- coding: utf-8 -*-
# @Author : maxin-nk
# @Email : xxin_ma@163.com
# @File : self_attention

import torch
import math
import torch.nn as nn
import pickle
import numpy as np


class SelfAttention(nn.Module):
    """
    自注意力机制模块：
    1.Q,K,V构造
    2.线性变换
    3.softmax
    """
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        # nn.Linear(in_features, out_features, bias):对输入数据应用线性变换：:math:y = xA^T + b
        self.Q_W = nn.Linear(embed_size, embed_size, bias=False)
        self.K_W = nn.Linear(embed_size, embed_size, bias=False)
        self.V_W = nn.Linear(embed_size, embed_size, bias=False)


    def forward(self, query, key, value, mask=None):
        # 由于encoder中attention的输入和decoder中attention输入不一样
        # 所以通过这种方式传参数
        # query, key, value: [batch_size, seq_len, embed_size]

        # 定义Q,K,V
        query = self.Q_W(query)
        key = self.K_W(key)
        value = self.V_W(value)

        # torch.einsum:对输入操作数（operands）在指定的维度上进行元素乘积的求和
        # query.shape[-1]**0.5: dk=query矩阵列数开根号
        QK = torch.einsum('bqe,bke->bqk', query, key)/(query.shape[-1]**0.5)
        attention = torch.softmax(QK, dim=-1)

        out = torch.einsum('bqs,bse->bqe', attention, value)

        return out


class TransformerBlock(nn.Module):
    """
    Transformer Block:
    1.self_attention
    2.dropout(add & norm)
    3.feed_forward(升维+ReLU+降维)
    4.dropout(add & norm)

    """
    def __init__(self, embed_size, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size)

        # 为什么选择layerNorm？而不是BatchNorm?
        #
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # 点对点的前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),  # 升维
            nn.ReLU(),                                            # 非线性变换
            nn.Linear(embed_size*forward_expansion, embed_size),  # 降维
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):

        # step1.注意力
        attention_output = self.attention(query, key, value)

        # step2. add & norm
        # 执行skip connect,注意一定要和query相加，因为在encoder中query=key=value,但是在decoder中，query！=key,key=value
        attention_output = self.dropout(self.norm1(attention_output + query))

        # step3. feed_forward
        forward_output = self.feed_forward(attention_output)

        # step4. add & norm
        output = self.dropout(self.norm2(forward_output + attention_output))

        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    """
    def __init__(self, d_model, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.device = device

        # create matrix of [Seqlen, hiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(self.device)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]

        return x


class Encoder(nn.Module):
    """
    Encoder Layer:构建Transformer的encoder
    """
    def __init__(self, src_vocab_size=15000, embed_size=100, forward_expansion=4, dropout=0.1, num_layers=4, device='cpu', classification=False, output_size=None):
        # src_vocab_size:输出词的数量
        # embed_size:输出词嵌入的维度
        # forward_expansion: feed_forward升维的倍数
        # dropout: 过拟合处理
        # num_layer:一个encoder所包含的block块数
        # device: 模型训练设备
        # classificaiton:是否用于分类
        # output_size:输出词向量的维度
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, embed_size)
        self.device = device
        self.dropout = dropout

        # 直接使用list保存若干个block，后期通过for loop 进行数据的input以及output
        self.layers = [TransformerBlock(embed_size, dropout, forward_expansion) for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)

        self.classification = classification
        if classification:
            self.fc = nn.Linear(embed_size, output_size)

        self.pos_encoding = PositionalEncoding(d_model=embed_size, device=device)

    def forward(self, src):
        embed = self.embedding(src)
        embed = self.pos_encoding(embed)

        for block in self.layers:
            embed = block(embed, embed, embed)

        if self.classification:
            pooled_output = torch.einsum('bse->be', embed)
            return self.fc(pooled_output)

        return embed


if __name__ == '__main__':

    # encoder测试
    # permute(input, dims):返回原始张量 :attr:input 的一个视图，其维度经过置换。
    text = torch.LongTensor(np.zeros((64, 33)))
    label = torch.LongTensor(np.ones(64))
    # text, label = pickle.load(open('small_train_batch.pkl', 'rb'))
    text = text.permute(1, 0)
    print(text.shape)

    # embedding = nn.Embedding(15000, 100)
    # x = embedding(text)
    # att = SelfAttention(100)
    # out_2 = att(x, x, x)
    # print(out_2.shape)
    #
    # transformer_encoder = Encoder(forward_expansion=4, output_size=2, classification=True)
    # out_3 = transformer_encoder(text)
    # print(out_3.shape)

    # decoder测试


    print('hello world')
