# Final-PJ-NLP

对于BERT模型，首先将数据格式根据data_process_A&B.py将原始SemEval数据格式转化成json文件，之后运行train.py即可下载bert-large-uncased预训练权重进行分类任务

对于RoBERTa模型，根据data_process_A&B_roberta.py将原始SemEval数据处理，参照[fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.custom_classification.md)中使用RoBERTa在Custom Data上的操作处理成脚本所需的数据格式输入，具体训练操作可见[RoBERTa](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta)

本次实验的得到的所有权重在[Google Drive]()中可下载(待上传)。


## Reference
https://github.com/facebookresearch/fairseq

https://github.com/huggingface/transformers
