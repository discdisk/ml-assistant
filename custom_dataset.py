import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional

train_data = [["天气怎么样", "下雨了吗", "冷吗", "气温怎么样"],
              ["我今天要做的事", "我的记事表"], ["打开记事本", "打开编辑器"]]
train_data = [["开太阳了吗", "热不热"], ["等会儿的预定", "日程表"], ["编辑器", "开始编程"]]


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def data_split(dataset: List[Dict]):
    texts = []
    labels = []
    for d in dataset:
        texts.append(d['sentence'])
        labels.append(d['label'])
    return texts, labels


def get_dataset(data: Dict[str, List[str]], label_keys: Optional[List[str]] = None, model_name: str = "bert-base-chinese") -> Tuple[Dataset, List[str]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    label_keys = label_keys or list(set(data.keys()))
    dataset = [{'label': label_keys.index(k) if k in label_keys else len(
        label_keys), 'sentence': s} for k, v in data.items() for s in v]
    texts, labels = data_split(dataset)
    encodings = tokenizer(texts, truncation=True, padding=True)
    return CustomDataset(encodings, labels), label_keys
