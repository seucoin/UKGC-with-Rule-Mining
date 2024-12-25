# Uncertain Knowledge Graph Completion with Rule Mining
Source code for WISA-2024 paper: Uncertain Knowledge Graph Completion with Rule Mining.

Since KGs usually suffer from the problem of incompleteness, methods of rule mining and reasoning for knowledge graph completion are extensively studied due to their excellent interpretability. However, previous methods are all conducted under deterministic scenarios, neglecting the uncertainty of knowledge, making them unable to be directly applied to UKGs. In this paper, we propose a new framework on uncertain knowledge graph completion with rule mining. Our framework contains the following components: 1)**The Rule Mining Model** applies an encoder-decoder network transformer to take rule mining as a sequence-to-sequence task to generate rules. It models the uncertainty in UKGs and infer new triples by differentiable reasoning based on TensorLog with mined rules. 2)**The Confidence Prediction Model** uses a pre-trained language model to predict the triple confidence given the rules mined. 

![image](https://github.com/user-attachments/assets/ad7411f9-0f45-48c0-a320-c9b5575637b2)
## Requirement
**Step1** Create a virtual environment using `Anaconda` and enter it.

**Step2** Installing the following packages in the virtual environment:
```
pytorch == 2.1.1
transformers == 4.38.2
wandb == 0.16.1
```

## Datasets

We adopt CN15K and NL27K datasets to evaluate our models, UKRM and BCP. 

| Dataset   | #Entities  | #Relations | #Train   | #Valid | #Test  |
| --------- | ---------- | ---------- | -------- | ------ | ------ |
| CN15K     | 15,000     | 36         | 204,984  | 16,881 | 19,293 |
| NL27K     | 27,221     | 404        | 149,001  | 12,278 | 14,034 |

## Files
`bert-base-uncased` folder contains the BERT model downloaded from hugginface(https://huggingface.co/google-bert/bert-base-uncased) and it will be used in the confidence prediction model.\
`transformer` folder contains source codes for the rule mining model on uncertain knowledge graph (UKRM).\
`confidence_prediction.py` is the source code for confidence predcition model (BCP).\
`DATASET` folder contains datasets we used in our paper.\
`decode_rules` folder contains input preprocessed for the confidence prediction model. GLM-4 is used in the process so it is a little time-consuming and we offer the data can be used directly.

## Usage
To train the rule mining model, please run follow instruction:\
`python translate_train.py`\
To decode rules from the rule mining model, please run follow instruction:\
`python translate_decode.py`\
To run the confidence prediction model, please run follow instruction:\
`python confidence_predcition.py`


## Argument Descriptions

Here are explanations of some important args,

```bash
--data_path:      "path of kowledge graph"
--batch_size:     "batch size"
--d_word_vec:     "dimension of word vector"
--d_model:        "dimension of model (usually same wih d_word_vec)"
--d_inner:        "dimension of feed forward layer"
--n_layers:       "num of layers of both encoder and decoder"
--n_head:         "num of attention heads (needs to ensure tha d_k*n_head == d_model)"
--d_k:            "dimension of attention vector k"
--d_v:            "dimension of attention vector v (usually same with d_k)"
--dropout:        "probability of dropout"
--n_position:     "number of positions"
--lr_mul:         "learning rate multiplier"
--n_warmup_steps: "num of warmup steps"
--num_epoch:      "num of epochs"
--save_step:      "steps to save"
--decode_rule:    "decode_rule mode"
```

Configs are set in python files and in case you want to modify them. Normally, other args can be set to default values.


## Citation
Please cite our paper if you use SuperRL in your work.
```
@inproceedings{chen2024uncertain,
  title={Uncertain Knowledge Graph Completion with Rule Mining},
  author={Chen, Yilin and Wu, Tianxing and Liu, Yunchang and Wang, Yuxiang and Qi, Guilin},
  booktitle={International Conference on Web Information Systems and Applications},
  pages={100--112},
  year={2024},
  organization={Springer}
}
```