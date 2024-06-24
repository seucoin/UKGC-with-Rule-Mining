# Uncertain Knowledge Graph Completion with Rule Mining
This repo provides codes and data for the paper: Uncertain Knowledge Graph Completion with Rule Mining.

## Requirement
pytorch == 2.1.1\
transformers == 4.38.2\
wandb == 0.16.1

## Files
`bert-base-uncased` folder contains the BERT model downloaded from hugginface(https://huggingface.co/google-bert/bert-base-uncased) and it will be used in the confidence prediction model.\
`transformer` folder contains source codes for the rule mining model on uncertain knowledge graph (UKRM).\
`confidence_prediction.py` is the source code for confidence predcition model (BCP).\
`DATASET` folder contains datasets we used in our paper.\
`decode_rules` folder contains input preprocessed for the confidence prediction model. GLM-4 is used in the process so it is a little time-consuming and we offer the data can be used directly.

## Usage
To train the rule mining model, you just need to run:\
`python translate_train.py`\
To decode rules from the rule mining model, you just need to run:\
`python translate_decod.py`\
To run the confidence prediction model, you just need to run:\
`python confidence_predcition.py`

## Config
Configs are set in python files and in case you want to modify them, here is a description.
```
"data_path": path of kowledge graph
"batch_size": batch size
"d_word_vec": dimension of word vector
"d_model": dimension of model (usually same wih d_word_vec)
"d_inner": dimension of feed forward layer
"n_layers": num of layers of both encoder and decoder
"n_head": num of attention heads (needs to ensure tha d_k*n_head == d_model)
"d_k": dimension of attention vector k
"d_v": dimension of attention vector v (usually same with d_k)
"dropout": probability of dropout
"n_position": number of positions
"lr_mul": learning rate multiplier
"n_warmup_steps": num of warmup steps
"num_epoch": num of epochs
"save_step": steps to save
"decode_rule": decode_rule mode
```