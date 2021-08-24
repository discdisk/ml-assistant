from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pprint import pprint
import json
from os import path
from pathlib import Path
current_path = Path(__file__).resolve().parent

torchscript = False
model_name = f"{current_path}/results/checkpoint-280"
test_data = ["开太阳了吗", "热不热", "等会儿的预定", "日程表", "编辑器", "开始编程",
             "VS code", "要带雨伞吗", "输入", "写点什么", "做点笔记", "坐飞机", "打酱油", "我想等会儿去看电影", "不知道外面下雨了没", "决赛真的有意思"]


# load label keys
if path.isfile(f'{current_path}/label_keys.json'):
    with open(f'{current_path}/label_keys.json') as json_file:
        label_keys = json.load(json_file)
        label_keys.append('None')
else:
    print(f'{current_path}/label_keys.json file missing!!')
    exit()

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')


def main():
    pt_batch = tokenizer(test_data, padding=True, truncation=True,
                         max_length=512, return_tensors="pt")
    if not torchscript:
        pt_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=False, torchscript=torchscript)
        pt_outputs = pt_model(**pt_batch)
        pt_outputs_digits = pt_outputs.logits.softmax(dim=1)
        pprint(pt_outputs_digits)
        pt_outputs = pt_outputs_digits.argmax(dim=1)
        pprint([f'{t}: {label_keys[label]} {digit[label]:.2f}' for t, label,
                digit in zip(test_data, pt_outputs, pt_outputs_digits)])
    else:
        # Creating a dummy input
        tokens_tensor = torch.tensor(pt_batch['input_ids'])
        att_tensor = torch.tensor(pt_batch['attention_mask'])
        segments_tensors = torch.tensor(pt_batch['token_type_ids'])
        dummy_input = [tokens_tensor, segments_tensors]

        # Creating the trace
        # load and save with torch jit
        pt_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=False, torchscript=torchscript)
        traced_model = torch.jit.trace(
            pt_model, [tokens_tensor, segments_tensors])
        torch.jit.save(traced_model, f'{current_path}/traced_bert.pt')

        # load torch jit model
        loaded_model = torch.jit.load(f'{current_path}/traced_bert.pt')
        loaded_model.eval()

        pooled_output, encoder_layer = loaded_model(*dummy_input)

        pooled_output_digits = pooled_output.softmax(dim=1)
        pooled_output = pooled_output.argmax(dim=1)

        pprint([f'{t}: {label_keys[label]} {digit[label]:.2f}' for t, label,
                digit in zip(test_data, pooled_output, pooled_output_digits)])


def get_model():
    if not torchscript:
        loaded_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=False, torchscript=torchscript)
    else:
        loaded_model = torch.jit.load(f'{current_path}/traced_bert.pt')
    loaded_model.eval()
    return loaded_model, tokenizer, label_keys


if __name__ == '__main__':
    main()
