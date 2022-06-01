import random
import pandas as pd
import json

templates = ["{}", "If the following statement is in common sense?[SEP]{}","If '{}' is in common sense?"]
template_id = 2
train_data = dict()
train_label = dict()
train_data_ = pd.read_csv("SemEval/ALL data/Training  Data/subtaskA_data_all.csv", sep='\t')[["id","sent0",'sent1']]
train_data_.to_csv("SemEval/ALL data/Training  Data/subtaskA_data_all.csv", sep="\t", index=False)
with open("SemEval/ALL data/Training  Data/subtaskA_data_all.csv", "r") as f:
    train_data_ = f.readlines()[1:]
    for item in train_data_:
        id, sent0, sent1 = item.strip().split("\t")
        train_data[str(id)] = [sent0, sent1]

with open("SemEval/ALL data/Training  Data/subtaskA_answers_all.csv", "r") as f:
    train_label_ = f.readlines()
    for item in train_label_:
        id, false_sent_id = item.strip().split(",")
        train_label[id] = int(false_sent_id)
    
train_sample = []
count = 0
for id, sents in train_data.items():
    false_id = train_label[id]
    true_id = 0 if false_id else 1
    train_sample.append({"id":count,"sentence":templates[template_id].format(sents[true_id]),"label":1})
    count += 1
    train_sample.append({"id":count, "sentence":templates[template_id].format(sents[false_id]),"label":0})
    count += 1

with open("SemEval/TaskA/temp2/train.json", "w") as f:
    json.dump(train_sample, fp=f, indent=4)


# 处理dev set:
train_data = dict()
train_label = dict()
train_data_ = pd.read_csv("SemEval/ALL data/Dev Data/subtaskA_dev_data.csv", sep='\t')
train_data_.to_csv("SemEval/ALL data/Dev Data/subtaskA_dev_data.csv", sep="\t", index=False)
with open("SemEval/ALL data/Dev Data/subtaskA_dev_data.csv", "r") as f:
    train_data_ = f.readlines()[1:]
    for item in train_data_:
        id, sent0, sent1 = item.strip().split("\t")
        train_data[str(id)] = [sent0, sent1]

with open("SemEval/ALL data/Dev Data/subtaskA_gold_answers.csv", "r") as f:
    train_label_ = f.readlines()
    for item in train_label_:
        id, false_sent_id = item.strip().split(",")
        train_label[id] = int(false_sent_id)
    
train_sample = []
count = 0
for id, sents in train_data.items():
    false_id = train_label[id]
    true_id = 0 if false_id else 1
    train_sample.append({"id":count,"sentence":templates[template_id].format(sents[true_id]),"label":1})
    count += 1
    train_sample.append({"id":count, "sentence":templates[template_id].format(sents[false_id]),"label":0})
    count += 1

with open("SemEval/TaskA/temp2/dev.json", "w") as f:
    json.dump(train_sample, fp=f, indent=4)
