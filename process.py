from sklearn.model_selection import train_test_split
import tokenizers
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot  as plt
import torch
import random
pd.options.mode.chained_assignment = None  # default='warn'

RANDOM_SEED = 100

# torch.cuda.current_device()
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    data = pd.read_csv("./Dataset/datas/cleaned_data.csv", encoding='utf-8')
    # task1
    data['labels'] = data[data.columns[4:10]].values.tolist()
    data = data[['text', 'labels']]
    # task2
    # data['labels'] = data['sentiment']
    # data = data.rename(columns={'sentiment': 'labels'})


    # 查看句子长度分布
    # field_length = data.text.astype(str).map(len)
    # sns.displot(field_length.tolist())
    # plt.xlim([0, 256])
    # plt.xlabel('Token count')
    # plt.show()

    # 中文分词
    tokenizer = tokenizers.hand_tokenizer()
    data['text'] = tokenizer.cut(data.text)

    df_train, df_test = train_test_split(
        data,
        test_size=0.2,
        random_state=RANDOM_SEED
    )
    df_val, df_test = train_test_split(
        df_test,
        test_size=0.5,
        random_state=RANDOM_SEED
    )


    word2idx = tokenizer.build_vocab(df_train.text, 10000)

    df_train['text'] = tokenizer.encoder(df_train.text, word2idx, 125)
    df_val['text'] = tokenizer.encoder(df_val.text, word2idx, 125)
    df_test['text'] = tokenizer.encoder(df_test.text, word2idx, 125)

    train_data_loader = tokenizer.create_data_loader(df_train, True, 125, 64)

    print(next(iter(train_data_loader)).shape())
