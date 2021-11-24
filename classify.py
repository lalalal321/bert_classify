# encoding: utf-8
"""
@author: zhou
@time: 2021/11/24 10:31
@file: classify.py
@desc: 
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

MODEL_NAME = "./bert-medium"
NUM_LABELS = 4
MAX_LEN = 1500
BATCH_SIZE = 16
EPOCHS = 10
label_dict = {}
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def get_learnable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


class MyDataset(Dataset):
    # 读取
    def __init__(self, mode, tokenizer, df_data):
        assert mode in ["train", "test"]  # dev set
        self.mode = mode
        self.df = df_data
        self.len = len(self.df)
        self.tokenizer = tokenizer  # BERT tokenizer

    def __getitem__(self, idx):
        if self.mode == "test":
            text, label = self.df.iloc[idx, 3:5].to_numpy()
            label_tensor = torch.tensor(label)
        else:
            text, label = self.df.iloc[idx, 4:6].to_numpy()
            # 将label转为 tensor
            label_tensor = torch.tensor(label)

        # 建立 BERT tokens 并加入分隔符[SEP]
        tokens_text = self.tokenizer.tokenize(text)
        word_pieces = ["[CLS]"] + tokens_text + ["[SEP]"]
        len_text = len(word_pieces)

        # 将token序列转换成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        # 不使用segement_tensor
        # segments_tensor = torch.tensor([0] * len_query + [1] * len_text, dtype=torch.long)
        # print("size",tokens_tensor.shape)

        return (tokens_tensor, label_tensor)

    def __len__(self):
        return self.len


def my_label(ipc_label):
    return label_dict[ipc_label]


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    # segments_tensors = [s[1] for s in samples]

    # 测试集有 labels
    if samples[0][1] is not None:
        label_ids = torch.stack([s[1] for s in samples])
    else:
        label_ids = None

    # zero pad 到同一序列长度
    tokens_tensors = pad_sequence(tokens_tensors,
                                  batch_first=True)
    # segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    # padding mask，将同一个batch中的序列padding到同一长度
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    # print(tokens_tensors.shape)
    return tokens_tensors, masks_tensors, label_ids


# 模型测试
def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0

    model.eval()  # 推理模式
    with torch.no_grad():
        for data in dataloader:
            tokens_tensors, masks_tensors, labels = [t.to(device) for t in data]

            logits = model(input_ids=tokens_tensors, attention_mask=masks_tensors).logits
            _, pred = torch.max(logits.data, 1)
            # print(labels.shape, pred.shape, tokens_tensors.shape, masks_tensors.shape)

            # 分类准确率计算
            if compute_acc:
                total += labels.size(0)
                correct += (pred == labels).sum().item()
            # 将当前 batch 记录下来
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
            # print((pred == labels).sum().item())
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions


if __name__ == '__main__':
    df_data = pd.read_csv("./data/train_B23.csv")
    df_test_data = pd.read_csv("./data/test_B23.csv")

    # 标签
    labels = df_test_data["IPC"].unique()
    NUM_LABELS = len(labels)
    count = 0
    for label in labels:
        label_dict[label] = count
        count += 1

    df_data["length"] = df_data["title"].str.len() + df_data["abstract"].str.len()
    df_data["text"] = df_data["title"] + df_data["abstract"]
    df_data["label"] = df_data.IPC.apply(my_label)

    df_test_data["text"] = df_test_data["title"] + df_test_data["abstract"]
    df_test_data["label"] = df_test_data.IPC.apply(my_label)

    df_data = shuffle(df_data)

    # len > 1500 的data截断
    df_data["text"] = df_data["text"].apply(lambda x: x[:MAX_LEN] if len(x) > MAX_LEN else x)
    df_test_data["text"] = df_test_data["text"].apply(lambda x: x[:MAX_LEN] if len(x) > MAX_LEN else x)

    train_set = MyDataset("train", tokenizer=tokenizer, df_data=df_data)
    test_set = MyDataset("test", tokenizer=tokenizer, df_data=df_test_data)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

    # 模型相关
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model_params = get_learnable_params(model)
    optimizer = torch.optim.Adam(model_params, lr=5e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    start = time.time()
    train_acc = []

    for epoch in range(EPOCHS):

        running_loss = 0.0
        step, steps = 0, len(df_data) // BATCH_SIZE
        for data in train_loader:
            tokens_tensors, masks_tensors, labels = [t.to(device) for t in data]

            optimizer.zero_grad()
            # print(tokens_tensors.shape,masks_tensors.shape, labels.shape)
            # forward pass
            outputs = model(input_ids=tokens_tensors, attention_mask=masks_tensors, labels=labels)
            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()

            print("Batch: [{}/{}], Loss: {}".format(step, steps, loss.item()))

            # 记录当前的 batch loss
            running_loss += loss.item()
            step += 1
        # 计算准确率
        _, acc = get_predictions(model, test_loader, compute_acc=True)
        train_acc.append(acc)

        print(f"batch size:{BATCH_SIZE}")
        print(f'[epoch {epoch + 1}] loss: {running_loss:3f}, acc: {acc:3f}')

    end = time.time()
    print(f"time:{end - start:.2f}")
    plt.plot([i + 1 for i in range(EPOCHS)], train_acc)
