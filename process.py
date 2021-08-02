import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
from transformers import AdamW, BertTokenizer

import tokenizers_util
import model
import train

pd.options.mode.chained_assignment = None  # default='warn'

RANDOM_SEED = 100

# torch.cuda.current_device()
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    set_seed(100)

    data = pd.read_csv("./Dataset/datas/cleaned_data.csv", encoding='utf-8')
    # task1
    data['labels'] = data[data.columns[4:10]].values.tolist()
    data = data[['text', 'labels']]
    # task2
    # data = data.rename(columns={'sentiment': 'labels'})
    # data = data[['text', 'labels']]
    # print(data['labels'].value_counts())

    # 查看句子长度分布
    # field_length = data.text.astype(str).map(len)
    # sns.displot(field_length.tolist())
    # plt.xlim([0, 256])
    # plt.xlabel('Token count')
    # plt.show()

    # 中文分词
    tokenizer = tokenizers_util.hand_tokenizer()
    data['text'] = tokenizer.cut(data.text)
    data['text'].replace('', np.nan, inplace=True)
    data = data.dropna(subset=['text'])

    df_train, df_test = train_test_split(
        data,
        test_size=0.2,
        random_state=RANDOM_SEED
    )
    df_val, df_test = train_test_split(
        df_test,
        test_size=0.1,
        random_state=RANDOM_SEED
    )

    word2idx = tokenizer.build_vocab(df_train.text, 10000)

    embeddings = tokenizer.load_pretrained_vectors(word2idx, 'Dataset/pre_train_vec/sgns.wiki.bigram-char')
    embeddings = torch.tensor(embeddings)

    # 均衡训练集
    # class_counts = df_train['labels'].value_counts().values  # dataset has 10 class-1 samples, 1 class-2 samples, etc.
    # num_samples = len(df_train.labels)
    # labels = df_train.labels.tolist()
    # class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
    # weights = [class_weights[int(labels[i])] for i in range(num_samples)]
    # sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

    # train_data_loader = tokenizer.create_data_loader(df_train, word2idx, True, 125, 64)
    # train_data_loader = tokenizer.create_data_loader(df_train, word2idx, True, 125, 64, sampler=sampler)
    # val_data_loader = tokenizer.create_data_loader(df_val, word2idx, True, 125, 64)
    # test_data_loader = tokenizer.create_data_loader(df_test, word2idx, True, 125, 64)

    train_data_loader = tokenizer.create_data_loader(df_train, word2idx, True, 125, 1)
    # train_data_loader = tokenizer.create_data_loader(df_train, word2idx, True, 125, 1, sampler=sampler)
    val_data_loader = tokenizer.create_data_loader(df_val, word2idx, True, 125, 1)
    test_data_loader = tokenizer.create_data_loader(df_test, word2idx, True, 125, 1)

    # label_nums = [0, 0, 0, 0, 0, 0]  # 二分类
    # for num, batch in enumerate(train_data_loader):
    #     input_ids = batch["input_ids"]
    #     targets = batch["targets"]
    #     for target_x in targets:
    #         for target_i in range(len(target_x)):
    #             lable = target_x[target_i].item()
    #             label_nums[target_i] += int(lable)
    # print("dddd", label_nums)

    # TextCNN 预训练，finetune
    # model = model.TextCNN(pretrained_embedding=embeddings,
    #                       freeze_embedding=True, num_classes=6
    #                       )

    # # TextCNN 自训练
    # model = model.TextCNN(vocab_size=len(word2idx), num_classes=6)

    # LSTM 预训练
    # model = model.LSTM(pretrained_embedding=embeddings,
    #                    freeze_embedding=True,
    #                    num_classes=6
    # )

    # Bi-LSTM 自训练
    # model = model.Bi_LSTM(vocab_size=len(word2idx), num_classes=6)

    # Bi-LSTM 预训练
    # model = model.Bi_LSTM(pretrained_embedding=embeddings,
    #                       freeze_embedding=True,
    #                       num_classes=6
    #                       )

    # RCNN
    model = model.RCNN(pretrained_embedding=embeddings,
                       freeze_embedding=False,
                       num_classes=6
                       )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # ----- bert
    # MODEL_NAME = 'hfl/chinese-bert-wwm-ext'

    # bert_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # train_data_loader = tokenizers_util.bert_data_loader(df_train, True, bert_tokenizer, 125, 64, sampler)
    # val_data_loader = tokenizers_util.bert_data_loader(df_val, True, bert_tokenizer, 125, 64)

    # model = model.My_BertModel(MODEL_NAME)

    # optimizer = AdamW(model.parameters(), lr=0.001)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # ----- bert

    model.to(device)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()

    # train.train_epoch(model, optimizer, scheduler, device, loss_fn, train_data_loader, val_data_loader, 30)
    train.train_epoch(model,
                      optimizer,
                      scheduler,
                      device,
                      loss_fn,
                      train_data_loader,
                      val_data_loader,
                      30,
                      is_lstm=True)

    # print("结束完事")
    #
    # lr_mult = (1 / 1e-5) ** (1 / 100)
    # lr = []
    # losses = []
    # best_loss = 1e9
    # for batch in train_data_loader:
    #     input_ids = batch["input_ids"].to(device)
    #     targets = batch["targets"].to(device)
    #     # attention_mask = batch["attention_mask"].to(device)
    #     data = input_ids
    #     label = targets
    #     # forward
    #     out = model(input_ids=input_ids
    #                 # attention_mask=attention_mask
    #                 )
    #     loss = loss_fn(out, label)
    #     # backward
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
    #     losses.append(loss.item())
    #     for param_group in optimizer.param_groups:  # 在每次更新参数前迭代更改学习率
    #         param_group["lr"] = optimizer.state_dict()['param_groups'][0]['lr'] * lr_mult
    #
    #     if loss.item() < best_loss:
    #         best_loss = loss.item()
    #     if loss.item() > 4 * best_loss or optimizer.state_dict()['param_groups'][0]['lr'] > 1.:
    #         break
    #
    # plt.figure()
    # plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
    # plt.xlabel('learning rate')
    # plt.ylabel('loss')
    # plt.plot(np.log(lr), losses)
    # plt.show()
    # plt.figure()
    # plt.xlabel('num iterations')
    # plt.ylabel('learning rate')
    # plt.plot(lr)
    #
    # print('O 了')
