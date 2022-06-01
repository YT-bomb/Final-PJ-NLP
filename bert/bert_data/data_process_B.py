import random
import pandas as pd
import json


templates = ["{}[SEP]{}", "{} is against common sense because {}","If {} is in common sense. [SEP] {} is against common sense because {}"]
template_id = 0
train_data = dict()
train_data_ = pd.read_csv("SemEval/ALL data/Training  Data/subtaskB_data_all.csv")

answer2idx = {"A":0, "B":1, "C":2}
for i in range(len(train_data_)):
    id, FalseSent, Oa, Ob, Oc = train_data_.loc[i,:]
    train_data[str(id)] = {
        "FalseSent": FalseSent,
        "Option": [Oa, Ob, Oc]
    }

with open("SemEval/ALL data/Training  Data/subtaskA_answers_all.csv", "r") as f:
    train_label_ = f.readlines()
    for item in train_label_:
        id, false_sent_id = item.strip().split(",")
        train_data[str(id)]["false_id"] = int(false_sent_id)

with open("SemEval/ALL data/Training  Data/subtaskA_data_all.csv", "r") as f:
    train_data_ = f.readlines()[1:]
    for item in train_data_:
        id, sent0, sent1 = item.strip().split("\t")
        train_data[str(id)]["TrueSent"] = sent0 if train_data[str(id)]["false_id"] else sent1

with open("SemEval/ALL data/Training  Data/subtaskB_answers_all.csv", "r") as f:
    train_label_ = f.readlines()
    for item in train_label_:
        id, reason = item.strip().split(",")
        train_data[str(id)]["Reason"] = answer2idx[reason]
    
train_sample = []

for id, data in train_data.items():
    if template_id != 2:
        train_sample.append({"sentence":templates[template_id].format(data["FalseSent"], data["Option"][data["Reason"]]),"label":1})
        for i in ({0,1,2} - {data["Reason"]}):
            train_sample.append({"sentence":templates[template_id].format(data["FalseSent"], data["Option"][i]),"label":0})
    else:
        train_sample.append({"sentence":templates[template_id].format(data["TrueSent"], data["FalseSent"], data["Option"][data["Reason"]]),"label":1})
        for i in ({0,1,2} - {data["Reason"]}):
            train_sample.append({"sentence":templates[template_id].format(data["TrueSent"], data["FalseSent"], data["Option"][i]),"label":0})

with open("SemEval/TaskB/data/train.json", "w") as f:
    json.dump(train_sample, fp=f, indent=4)


train_data = dict()
train_data_ = pd.read_csv("SemEval/ALL data/Dev Data/subtaskB_dev_data.csv")
for i in range(len(train_data_)):
    id, FalseSent, Oa, Ob, Oc = train_data_.loc[i,:]
    train_data[str(id)] = {
        "FalseSent": FalseSent,
        "Option": [Oa, Ob, Oc]
    }

with open("SemEval/ALL data/Dev Data/subtaskA_gold_answers.csv", "r") as f:
    train_label_ = f.readlines()
    for item in train_label_:
        id, false_sent_id = item.strip().split(",")
        train_data[str(id)]["false_id"] = int(false_sent_id)

with open("SemEval/ALL data/Dev Data/subtaskA_dev_data.csv", "r") as f:
    train_data_ = f.readlines()[1:]
    for item in train_data_:
        id, sent0, sent1 = item.strip().split("\t")
        train_data[str(id)]["TrueSent"] = sent0 if train_data[str(id)]["false_id"] else sent1

with open("SemEval/ALL data/Dev Data/subtaskB_gold_answers.csv", "r") as f:
    train_label_ = f.readlines()
    for item in train_label_:
        id, reason = item.strip().split(",")
        train_data[str(id)]["Reason"] = answer2idx[reason]
    
train_sample = []

for id, data in train_data.items():
    if template_id != 2:
        train_sample.append({"sentence":templates[template_id].format(data["FalseSent"], data["Option"][data["Reason"]]),"label":1})
        for i in ({0,1,2} - {data["Reason"]}):
            train_sample.append({"sentence":templates[template_id].format(data["FalseSent"], data["Option"][i]),"label":0})
    else:
        train_sample.append({"sentence":templates[template_id].format(data["TrueSent"], data["FalseSent"], data["Option"][data["Reason"]]),"label":1})
        for i in ({0,1,2} - {data["Reason"]}):
            train_sample.append({"sentence":templates[template_id].format(data["TrueSent"], data["FalseSent"], data["Option"][i]),"label":0})
random.shuffle(train_sample)
with open("SemEval/TaskB/data/dev.json", "w") as f:
    json.dump(train_sample, fp=f, indent=4)
