from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from custom_dataset import get_dataset
import json
from os import path
train_data = {'天气': ["天气怎么样", "下雨了吗", "冷吗", "气温怎么样"],
              '预定': ["我今天要做的事", "我的记事表"], '编辑器': ["打开记事本", "打开编辑器"], '吃饭': ["吃个西瓜", "打篮球"]}
test_data = {'天气': ["开太阳了吗", "热不热"], '预定': [
    "等会儿的预定", "日程表"], '编辑器': ["编辑器", "开始编程"]}

if path.isfile('label_keys.json'):
    with open('label_keys.json') as json_file:
        label_keys = json.load(json_file)
else:
    label_keys = None

train_dataset, label_keys = get_dataset(train_data, label_keys=label_keys)
test_dataset, label_keys = get_dataset(test_data, label_keys=label_keys)

with open('label_keys.json', 'w') as json_file:
    json.dump(label_keys, json_file)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese", num_labels=len(label_keys)+1)

batch_size = 1
training_args = TrainingArguments(
    output_dir='./results',                  # output directory
    num_train_epochs=40,                      # total number of training epochs
    # batch size per device during training
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,   # batch size for evaluation
    # number of warmup steps for learning rate scheduler
    warmup_steps=80,
    weight_decay=0.01,                       # strength of weight decay
    logging_dir='./logs',                    # directory for storing logs
    logging_steps=20,
    save_steps=40,
    evaluation_strategy='steps',
    save_strategy='steps',
    logging_strategy='steps',
    eval_steps=20,
    label_smoothing_factor=0.1
)

trainer = Trainer(
    # the instantiated 🤗 Transformers model to be trained
    model=model,
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset            # evaluation dataset
)

trainer.train()
