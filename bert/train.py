from cProfile import label
from sklearn.model_selection import learning_curve
from transformers.models.bert.modeling_bert import BertPreTrainedModel,BertModel
from transformers import BertTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, TensorDataset)

from transformers.optimization import (AdamW, get_linear_schedule_with_warmup)
import os
import json
import torch

import copy
import errno
import tqdm
from typing import List, Optional
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * 2, self.config.num_labels)
                )

        # Initialize weights and apply final processing
        self.post_init()

    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        # if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class DATASET(Dataset):
    def __init__(self, datafile, tokenizer_path="bert-large-uncased") -> None:
        super(Dataset).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.max_seq_length = 128
        self.data = json.load(open(datafile, "r"))

    
    def __getitem__(self, index):
        data = self.data[index]
        label = torch.tensor(data['label'], dtype=torch.long)
        input_ids_a, input_mask_a, segment_ids_a = self.tensorize(data["sentence"])
        return index, (input_ids_a, input_mask_a, segment_ids_a, label)
    
    def tensorize(self, text, cls_token_segment_id=0, pad_token_segment_id=0, sequence_a_segment_id=0):
        tokens_a = self.tokenizer.tokenize(text)
        num_extra_tokens = 2
        if len(tokens_a) > self.max_seq_length - num_extra_tokens: # edited here to make it for sequence length == 68
            tokens_a = tokens_a[:(self.max_seq_length - num_extra_tokens)]
        
        seq_tokens_a = [self.tokenizer.cls_token] + tokens_a #  + [self.tokenizer.sep_token]
        input_ids_a = self.tokenizer.convert_tokens_to_ids(seq_tokens_a) + [self.tokenizer.vocab[self.tokenizer.sep_token]]
        segment_ids_a = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        input_mask_a = [1] * len(input_ids_a)
        seq_len_a = len(input_ids_a)
        seq_padding_len_a = self.max_seq_length - seq_len_a
        input_ids_a += seq_padding_len_a * [0,]
        input_mask_a += seq_padding_len_a * [0,]
        segment_ids_a += seq_padding_len_a * [pad_token_segment_id,]

        input_ids_a = torch.tensor(input_ids_a, dtype=torch.long)
        input_mask_a = torch.tensor(input_mask_a, dtype=torch.long)
        segment_ids_a = torch.tensor(segment_ids_a, dtype=torch.long)
        return (input_ids_a, input_mask_a, segment_ids_a)

    def __len__(self):
        return len(self.data)

def evaluate(num_workers, eval_batch_size, device, eval_dataset, model):
    # eval_examples, eval_features, na_prob_thresh=1.0, pred_only=False):
    all_results = []
    model.eval()
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, num_workers=num_workers, sampler=eval_sampler, batch_size=eval_batch_size)

    # get predictions

    pred_labels = []
    golden_labels = []
    with torch.no_grad():
        for step, (indexes, batch) in enumerate(eval_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids":       batch[0],
                "attention_mask":  batch[1],
                "position_ids":    batch[2]
            }
            outputs = model(**inputs)
            pred_logits = outputs[0]

            golden_labels.extend(batch[3].cpu().tolist())
            _, pre_labels = torch.max(pred_logits, 1)
            pred_labels.extend(pre_labels.tolist())
    positive = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == golden_labels[i]:
            positive += 1
        
    acc = positive/len(pred_labels)
    return acc

def main():
    trainfile = "SemEval/TaskA/data/dev.json"
    devfile = "SemEval/TaskA/data/train.json"
    num_workers = 8
    batch_size = 32
    dev_batch_size = 32
    epochs = 10
    learning_rate = 1e-4
    device = "cuda:0"
    warmup_steps = 100
    output_dir = "trained_model"

    mkdir(output_dir)
    
    trainsets = DATASET(trainfile)
    devsets = DATASET(devfile)
    train_sampler = RandomSampler(trainsets)
    train_dataloader = DataLoader(trainsets, num_workers=num_workers, sampler=train_sampler, batch_size=batch_size) #, collate_fn=trim_batch)

    config = BertConfig.from_pretrained("bert-large-uncased")
    config.num_labels = 2
    model = BertForSequenceClassification(config)
    model.bert = model.bert.from_pretrained("bert-large-uncased")
    model.zero_grad()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    t_total = len(train_dataloader) * epochs
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)
    best_model = {
        'epoch': 0,
        'model': copy.deepcopy(model), #model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    best_acc = {"train":0, "dev":0}
    print("Begin Training!")
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        iters = 0
        for step, (indexes, batch) in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids":       batch[0],
                "attention_mask":  batch[1],
                "position_ids":    batch[2],
                "labels":          batch[3]
            }
            outputs = model.forward(**inputs)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()
            iters += 1
        
        dev_acc = evaluate(num_workers, dev_batch_size, device, devsets, model)
        train_acc = evaluate(num_workers, dev_batch_size, device, trainsets, model)

        print("Epoch: {}---Average Loss: {}---Train Acc: {}---Dev Acc: {}".format(epoch, total_loss/iters, train_acc, dev_acc))
        
        if dev_acc > best_acc["dev"]:
            best_acc["dev"] = dev_acc
            best_acc["train"] = train_acc
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)

        output_dir_check = os.path.join(output_dir, 'checkpoint-{}'.format(epoch))
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

        save_num = 0
        while (save_num < 10):
            try:
                print("Saving model attempt: {}".format(save_num))
                model_to_save.save_pretrained(output_dir_check)
                break
            except:
                save_num += 1
    
    output_dir = os.path.join(output_dir, 'best-{}'.format(best_model["epoch"]))
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

    save_num = 0
    while (save_num < 10):
        try:
            print("Saving model attempt: {}".format(save_num))
            model_to_save.save_pretrained(output_dir)
            break
        except:
            save_num += 1
    
    print("Best: Train Acc——{}; Dev Acc——{}".format(best_acc["train"], best_acc["dev"]))


if __name__ == "__main__":
    main() 

    