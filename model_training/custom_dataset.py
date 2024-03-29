import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional

train_data = {'天气': ["天气怎么样", "下雨了吗", "冷吗", "气温怎么样"],
              '预定': ["我今天要做的事", "我的记事表"],
              '编辑器': ["打开记事本", "打开编辑器"],
              '吃饭': ["吃个西瓜"],
              '后退': ['后退'],
              '床': ['床'],
              '鸟': ['鸟'],
              '猫': ['猫'],
              '狗': ['狗'],
              '下': ['下'],
              '跟随': ['跟随'],
              '向前': ['向前'],
              '去': ['去'],
              '快乐的': ['快乐的'],
              '房子': ['房子'],
              '学': ['学'],
              '左': ['左'],
              '不': ['不'],
              '关闭': ['关闭'],
              '右边': ['右边'],
              '停止': ['停止'],
              '树': ['树'],
              '向上': ['向上'],
              '视觉的': ['视觉的'],
              '是的': ['是的']}
test_data = {'天气': ["开太阳了吗", "热不热"],
             '预定': ["等会儿的预定", "日程表"],
             '编辑器': ["编辑器", "开始编程"]}


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


def get_tokenizer(model_name: str = "bert-base-chinese", use_fast: bool = False) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def get_dataset(data: Dict[str, List[str]], label_keys: Optional[List[str]] = None, model_name: str = "bert-base-chinese") -> Tuple[Dataset, List[str]]:
    tokenizer = get_tokenizer(model_name, use_fast=True)
    label_keys = label_keys or list(set(data.keys()))
    dataset = [{'label': label_keys.index(k) if k in label_keys else len(
        label_keys), 'sentence': s} for k, v in data.items() for s in v]
    texts, labels = data_split(dataset)
    encodings = tokenizer(texts, truncation=True, padding=True)
    return CustomDataset(encodings, labels), label_keys
