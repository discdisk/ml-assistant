from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pprint import pprint
import json
from os import path

if path.isfile('label_keys.json'):
    with open('label_keys.json') as json_file:
        label_keys = json.load(json_file)
        label_keys.append('None')
else:
    print('label_keys.json file missing!!')
    exit()
model_name = "./results/checkpoint-280"
pt_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, output_hidden_states=True, output_attentions=False)
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

texts = ["开太阳了吗", "热不热", "等会儿的预定", "日程表", "编辑器", "开始编程",
         "VS code", "要带雨伞吗", "输入", "写点什么", "做点笔记", "坐飞机", "打酱油", "我想等会儿去看电影", "不知道外面下雨了没", "决赛真的有意思"]
pt_batch = tokenizer(texts, padding=True, truncation=True,
                     max_length=512, return_tensors="pt")
pt_outputs = pt_model(**pt_batch)
pt_outputs_digits = pt_outputs.logits.softmax(dim=1)
pprint(pt_outputs_digits)
pt_outputs = pt_outputs_digits.argmax(dim=1)
pprint([f'{t}: {label_keys[label]} {digit[label]:.2f}' for t, label,
        digit in zip(texts, pt_outputs, pt_outputs_digits)])

# Creating the trace
print(pt_batch)
# Creating a dummy input
tokens_tensor = torch.tensor(pt_batch['input_ids'])
att_tensor = torch.tensor(pt_batch['attention_mask'])
segments_tensors = torch.tensor(pt_batch['token_type_ids'])
dummy_input = [tokens_tensor, segments_tensors]

pt_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, output_hidden_states=True, output_attentions=False, torchscript=True)
traced_model = torch.jit.trace(
    pt_model, [tokens_tensor, segments_tensors])
torch.jit.save(traced_model, "traced_bert.pt")

loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

pooled_output, encoder_layer = loaded_model(*dummy_input)

pooled_output_digits = pooled_output.softmax(dim=1)
pooled_output = pooled_output.argmax(dim=1)

pprint([f'{t}: {label_keys[label]} {digit[label]:.2f}' for t, label,
        digit in zip(texts, pooled_output, pooled_output_digits)])
