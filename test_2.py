# -*- coding: utf-8 -*-
# @Author : maxin-nk
# @Email : xxin_ma@163.com
# @File :
import torch

# 定义批次大小和序列长度
batch_size = 4
seq_length = 100

# 生成随机的源数据和目标数据
src_token_ids_batch = torch.randint(0, 1000, size=(batch_size, seq_length))
trg_token_ids_batch = torch.randint(0, 1000, size=(batch_size, seq_length))

# 假设填充值为 0（通常填充值是 0，但这取决于具体的实现）
src_pad_token = 0
trg_pad_token = 0

# 创建源数据的掩码
src_mask = (src_token_ids_batch != src_pad_token).unsqueeze(1).unsqueeze(2)

# 创建目标数据的填充掩码
trg_padding_mask = (trg_token_ids_batch != trg_pad_token).unsqueeze(1).unsqueeze(2)

# 创建未来单词掩码
trg_seq_length = trg_token_ids_batch.size(1)
trg_future_mask = torch.triu(torch.ones((trg_seq_length, trg_seq_length), dtype=torch.bool), diagonal=1)

# 结合填充掩码和未来单词掩码
trg_mask = trg_padding_mask & trg_future_mask.unsqueeze(0).unsqueeze(0)

print("Source mask:")
print(src_mask)
print("Target padding mask:")
print(trg_padding_mask)
print("Target future mask:")
print(trg_future_mask)
print("Combined target mask:")
print(trg_mask)


# # 创建一个全为 True 的上三角矩阵,将上三角矩阵的值反转为 False
# src_mask = torch.triu(torch.ones((batch_size, seq_length), dtype=torch.bool), diagonal=1).unsqueeze(1).unsqueeze(2)
# src_mask = ~ src_mask
#
# print(src_mask)
