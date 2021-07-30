import csv
import spacy_pkuseg as pkuseg
import pandas as pd
import numpy as np
import torch
import bz2
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset

class hand_DataSet(Dataset):
    def __init__(self, dataframe, with_label, max_len):
        self.input_ids = dataframe.text.to_numpy()
        self.with_label = with_label
        if with_label:
            self.targets = dataframe.labels.to_numpy()
        self.max_len = max_len

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        input_ids = self.input_ids[item]
        if self.with_label:
            return {
                'input_ids': torch.LongTensor(input_ids),
                'targets': torch.tensor(self.targets[item], dtype=torch.long)
            }
        else:
            return {
                'input_ids': torch.LongTensor(input_ids),
            }


class hand_tokenizer:

    def __init__(self):
        self.stop_words = pd.read_csv("./Dataset/stopwords/hit_stopwords.txt", quoting=csv.QUOTE_NONE,
                                      header=None, names=["text"], encoding='utf-8')

        self.tokenizer = pkuseg.pkuseg()

        self.space_tokenize = lambda x: x.split()



    def chinese_tokenizer(self, text_data):
        # 小写
        text_data = text_data.lower()
        # 分词
        text_data = list(self.tokenizer.cut(text_data))

        # 去停顿词
        text_data = [word.strip() for word in text_data if word not in self.stop_words.text.values]

        text_data = " ".join(text_data)

        return text_data

    def cut(self, dataframe):
        return dataframe.apply(self.chinese_tokenizer)

    def tfidf_vectorizer(self, text_data):
        datas = text_data.tolist()

        tfidf_model = TfidfVectorizer( max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 1), use_idf=1, smooth_idf=1, sublinear_tf=1)

        embedding = tfidf_model.fit_transform(datas)

        return embedding

    def build_vocab(self, text_data, max_size):

        word2idx = {}

        for sent in text_data:
            tokenized_sent = self.space_tokenize(sent)

            for token in tokenized_sent:
                word2idx[token] = word2idx.get(token, 0) + 1

        word_list = sorted([_ for _ in word2idx.items()], key=lambda x: x[1], reverse=True)[
                     :max_size]

        word2idx = {word_count[0]: idx for idx, word_count in enumerate(word_list)}

        word2idx.update({'<UNK>': len(word2idx), '<PAD>': len(word2idx) + 1})

        return word2idx

    def encoder(self, texts, word2idx, max_len):
        input_ids = []

        for sent in texts:

            tokenized_sent = self.space_tokenize(sent)
            sent_len = len(tokenized_sent)
            if sent_len < max_len:
                tokenized_sent += ['<PAD>'] * (max_len - sent_len)
            else:
                tokenized_sent = tokenized_sent[:max_len]

            input_id = [word2idx.get(token, word2idx.get('<UNK>')) for token in tokenized_sent]
            input_ids.append(input_id)

        return input_ids

    def load_pretrained_vectors(self, word2idx, fname):

        print("Loading pretrained vectors...")

        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())

        # Initilize random embeddings
        embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
        embeddings[word2idx['<PAD>']] = np.zeros((d,))

        # Load pretrained vectors
        count = 0
        for line in tqdm(fin):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in word2idx:
                count += 1
                embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

        print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

        return embeddings

    def create_data_loader(self, df, with_label, max_len, batch_size, sampler=None):
        ds = hand_DataSet(
            df,
            with_label,
            max_len
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,
            sampler=sampler
        )


class GPReviewDataset(Dataset):
    def __init__(self, dataframe, with_label, tokenizer, max_len):
        self.reviews = dataframe.text.to_numpy()
        self.with_label = with_label
        if with_label:
            self.targets = dataframe.labels.to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        if len(encoding['input_ids'].flatten()) == 151:
            print(review)
        if self.with_label:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(self.targets[item], dtype=torch.long)
            }
        else:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
            }


def bert_data_loader(df, with_label, tokenizer, max_len, batch_size, sampler=None):
    ds = GPReviewDataset(
        df,
        with_label,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        sampler=sampler
    )







