
# ===== Cell 1 =====
__author__ = "Amir Zur"

# ===== Cell 4 =====
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

# ===== Cell 5 =====
import os
import json 
import random
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score

from pyvene import CausalModel
# from pyvene.models.configuration_intervenable_model import RepresentationConfig

from pyvene.models.gpt2.modelings_intervenable_gpt2 import create_gpt2_lm
# from pyvene.models.gpt_neox.modelings_intervenable_gpt_neox import create_gpt_neox

# ===== Cell 6 =====
TRAIN_DIR = 'mqnli_factual'
DAS_DIR = 'mqnli_das'

# ===== Cell 7 =====
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# ===== Cell 9 =====
# JSON files generated from adapting MQNLI codebase
# https://github.com/atticusg/MultiplyQuantifiedData

class Hashabledict(dict):
    def __hash__(self):
        return hash(frozenset(self))

with open('tutorial_data/mqnli_q_projectivity.json') as f:
    determiner_signatures = json.load(f)
    determiner_signatures = Hashabledict({
        q1: Hashabledict({
            q2: Hashabledict({
                r1: Hashabledict({
                    r2: determiner_signatures[q1][q2][r1][r2]  
                    for r2 in determiner_signatures[q1][q2][r1]
                })  
                for r1 in determiner_signatures[q1][q2]
            })
            for q2 in determiner_signatures[q1]
        })
        for q1 in determiner_signatures
    })

with open('tutorial_data/mqnli_neg_signature.json') as f:
    negation_signature = Hashabledict(json.load(f))

with open('tutorial_data/mqnli_empty_signature.json') as f:
    emptystring_signature = Hashabledict(json.load(f))

with open('tutorial_data/mqnli_cont_signature.json') as f:
    compose_contradiction_signature = Hashabledict(json.load(f))

with open('tutorial_data/mqnli_neg_cont_signature.json') as f:
    compose_neg_contradiction_signature = Hashabledict(json.load(f))

# ===== Cell 11 =====
parents = {
    "N_P_O": [],
    "N_H_O": [],
    "N_O": ["N_P_O", "N_H_O"],
    "Adj_P_O": [],
    "Adj_H_O": [],
    "Adj_O": ["Adj_P_O", "Adj_H_O"],
    "NP_O": ["Adj_O", "N_O"],
    "Q_P_O": [],
    "Q_H_O": [],
    "Q_O": ["Q_P_O", "Q_H_O"],
    "V_P": [],
    "V_H": [],
    "V": ["V_P", "V_H"],
    "Adv_P": [],
    "Adv_H": [],
    "Adv": ["Adv_P", "Adv_H"],
    "VP": ["Adv", "V"],
    "QP_O": ["Q_O", "NP_O", "VP"],
    "Neg_P": [],
    "Neg_H": [],
    "Neg": ["Neg_P", "Neg_H"],
    "NegP": ["Neg", "QP_O"],
    "N_P_S": [],
    "N_H_S": [],
    "N_S": ["N_P_S", "N_H_S"],
    "Adj_P_S": [],
    "Adj_H_S": [],
    "Adj_S": ["Adj_P_S", "Adj_H_S"],
    "NP_S": ["Adj_S", "N_S"],
    "Q_P_S": [],
    "Q_H_S": [],
    "Q_S": ["Q_P_S", "Q_H_S"],
    "QP_S": ["Q_S", "NP_S", "NegP"]
}

# ===== Cell 12 =====
EMPTY = ""
IND = "independence"
EQV = "equivalence"
ENT = "entails"
REV = "reverse entails"
CON = "contradiction"
ALT = "alternation"
COV = "cover"
# all possible relation values from the original paper 
# (https://arxiv.org/pdf/1810.13033.pdf)
RELATIONS = [IND, EQV, ENT, REV, CON, ALT, COV]

Q_VALUES = [
    determiner_signatures[q1][q2]
    for q1 in determiner_signatures for q2 in determiner_signatures[q1]
]

values = {
    "N_P_O": ["tree", "rock"],
    "N_H_O": ["tree", "rock"],
    "N_O": [EQV, IND],
    "Adj_P_O": ["happy", "sad", EMPTY],
    "Adj_H_O": ["happy", "sad", EMPTY],
    "Adj_O": [EQV, IND, ENT, REV],
    "NP_O": [EQV, IND, ENT, REV],
    "Q_P_O": ["some", "every"],
    "Q_H_O": ["some", "every"],
    "Q_O": Q_VALUES,
    "V_P": ["climbed", "threw"],
    "V_H": ["climbed", "threw"],
    "V": [EQV, IND],
    "Adv_P": ["energetically", "joyfully", EMPTY],
    "Adv_H": ["energetically", "joyfully", EMPTY],
    "Adv": [EQV, IND, ENT, REV],
    "VP": [EQV, IND, ENT, REV],
    "QP_O": [EQV, IND, ENT, REV],
    "Neg_P": ["not", EMPTY],
    "Neg_H": ["not", EMPTY],
    "Neg": [
        negation_signature, 
        emptystring_signature, 
        compose_contradiction_signature, 
        compose_neg_contradiction_signature
    ],
    "NegP": RELATIONS,
    "N_P_S": ["child", "dog"],
    "N_H_S": ["child", "dog"],
    "N_S": [EQV, IND],
    "Adj_P_S": ["little", "cute", EMPTY],
    "Adj_H_S": ["little", "cute", EMPTY],
    "Adj_S": [EQV, IND, ENT, REV],
    "NP_S": [EQV, IND, ENT, REV],
    "Q_P_S": ["some", "every"],
    "Q_H_S": ["some", "every"],
    "Q_S": Q_VALUES,
    "QP_S": RELATIONS
}

# ===== Cell 13 =====
# adapted from original code for MQNLI:
# https://github.com/atticusg/MultiplyQuantifiedData/blob/master/natural_logic_model.py

def adj_merge(adj_p, adj_h):
    if adj_p == adj_h:
        return EQV
    if adj_p == EMPTY:
        return REV
    if adj_h == EMPTY:
        return ENT
    return IND

adv_merge = adj_merge

def noun_phrase(adj, noun):
    #merges a noun relation with an adjective relation
    #or a verb relation with an adverb relation
    # makes sense: if the objects are the same, then adjective's relation holds
    # otherwise, they're independent
    if noun == EQV:
        return adj
    return IND

verb_phrase = noun_phrase

def negation_merge(neg_p, neg_h):
    #merges negation
    if neg_p == neg_h and neg_p == EMPTY:
        return Hashabledict(emptystring_signature)
    if neg_p == neg_h and neg_p != EMPTY:
        return Hashabledict(negation_signature)
    if neg_p == EMPTY:
        return Hashabledict(compose_contradiction_signature)
    if neg_p != EMPTY:
        return Hashabledict(compose_neg_contradiction_signature)

negation_phrase = lambda neg, qp: neg[qp]

noun_merge = lambda n_p, n_h: EQV if n_p == n_h else IND
verb_merge = noun_merge

quantifier_merge = lambda q_p, q_h: determiner_signatures[q_p][q_h]

quantifier_phrase = lambda q, np, vp: q[np][vp]

functions = {
    "N_P_O": lambda: "tree",
    "N_H_O": lambda: "tree",
    "N_O": noun_merge,
    "Adj_P_O": lambda: "happy",
    "Adj_H_O": lambda: "happy",
    "Adj_O": adj_merge,
    "NP_O": noun_phrase,
    "Q_P_O": lambda: "some",
    "Q_H_O": lambda: "some",
    "Q_O": quantifier_merge,
    "V_P": lambda: "climbed",
    "V_H": lambda: "climbed",
    "V": verb_merge,
    "Adv_P": lambda: "energetically",
    "Adv_H": lambda: "energetically",
    "Adv": adv_merge,
    "VP": verb_phrase,
    "QP_O": quantifier_phrase,
    "Neg_P": lambda: "not",
    "Neg_H": lambda: "not",
    "Neg": negation_merge,
    "NegP": negation_phrase,
    "N_P_S": lambda: "dog",
    "N_H_S": lambda: "dog",
    "N_S": noun_merge,
    "Adj_P_S": lambda: "cute",
    "Adj_H_S": lambda: "cute",
    "Adj_S": adj_merge,
    "NP_S": noun_phrase,
    "Q_P_S": lambda: "some",
    "Q_H_S": lambda: "some",
    "Q_S": quantifier_merge,
    "QP_S": quantifier_phrase
}

# ===== Cell 14 =====
# coordinates to display the MQNLI tree
pos = {
    "N_P_O": (32, 0.3),
    "N_H_O": (30, 0.7),
    "N_O": (31, 1.3),
    "Adj_P_O": (28, 0),
    "Adj_H_O": (26, 0.5),
    "Adj_O": (27, 1),
    "NP_O": (29, 2),
    "Q_P_O": (24, 1.3),
    "Q_H_O": (22, 1.7),
    "Q_O": (23, 2.5),
    "V_P": (21, 0),
    "V_H": (19, 0.5),
    "V": (20, 1),
    "Adv_P": (17, -0.3),
    "Adv_H": (15, 0.2),
    "Adv": (16, 0.7),
    "VP": (18, 2),
    "QP_O": (25, 3),
    "Neg_P": (13, 2.5),
    "Neg_H": (11, 3),
    "Neg": (12, 3.5),
    "NegP": (14, 4),
    "N_P_S": (9, 2.2),
    "N_H_S": (7, 2.8),
    "N_S": (8, 3.3),
    "Adj_P_S": (5, 1.5),
    "Adj_H_S": (3, 2),
    "Adj_S": (4, 2.5),
    "NP_S": (6, 4.3),
    "Q_P_S": (2, 3.2),
    "Q_H_S": (0, 3.5),
    "Q_S": (1, 4),
    "QP_S": (10, 5)
}

# ===== Cell 15 =====
variables = list(parents.keys())  # pretty sure this preserves order?

# ===== Cell 16 =====
mqlni_model = CausalModel(variables, values, parents, functions, pos=pos)

# ===== Cell 21 =====
def print_premise(setting):
    print(
        setting["Q_P_S"],
        setting["Adj_P_S"],
        setting["N_P_S"],
        setting["Neg_P"],
        setting["Adv_P"],
        setting["V_P"],
        setting["Q_P_O"],
        setting["Adj_P_O"],
        setting["N_P_O"]
    )

def print_hypothesis(setting):
    print(
        setting["Q_H_S"],
        setting["Adj_H_S"],
        setting["N_H_S"],
        setting["Neg_H"],
        setting["Adv_H"],
        setting["V_H"],
        setting["Q_H_O"],
        setting["Adj_H_O"],
        setting["N_H_O"]
    )

# ===== Cell 27 =====
dataset = mqlni_model.generate_factual_dataset(100, sampler=mqlni_model.sample_input_tree_balanced, return_tensors=False)

X = [example['input_ids'] for example in dataset]
y = [example['labels'] for example in dataset]

# ===== Cell 28 =====
i = 0

print_premise(X[i])
print_hypothesis(X[i])
print(y[i]['QP_S'])

# ===== Cell 32 =====
config, tokenizer, model = create_gpt2_lm()

# ===== Cell 33 =====
def premise_to_string(setting):
    return \
        setting["Q_P_S"] + ' ' + \
        setting["Adj_P_S"] + ' ' + \
        setting["N_P_S"] + ' ' + \
        setting["Neg_P"] + ' ' + \
        setting["Adv_P"] + ' ' + \
        setting["V_P"] + ' ' + \
        setting["Q_P_O"] + ' ' + \
        setting["Adj_P_O"] + ' ' + \
        setting["N_P_O"]

def hypothesis_to_string(setting):
    return \
        setting["Q_H_S"] + ' ' + \
        setting["Adj_H_S"] + ' ' + \
        setting["N_H_S"] + ' ' + \
        setting["Neg_H"] + ' ' + \
        setting["Adv_H"] + ' ' + \
        setting["V_H"] + ' ' + \
        setting["Q_H_O"] + ' ' + \
        setting["Adj_H_O"] + ' ' + \
        setting["N_H_O"]

def preprocess_input(setting):
    return f'Premise: {premise_to_string(setting)}\nHypothesis: {hypothesis_to_string(setting)}\nRelation: '

def preprocess_output(setting):
    return setting['QP_S']

# ===== Cell 34 =====
IGNORE_INDEX = -100
MAX_LENGTH = 64

tokenizer.pad_token = tokenizer.eos_token

def preprocess(X, y):
    examples = [preprocess_input(x) for x in X]
    labels = [preprocess_output(y) for y in y]

    examples = tokenizer(
        examples, 
        padding='max_length', 
        max_length=MAX_LENGTH, 
        truncation=True, 
        return_tensors='pt'
    )
    labels = tokenizer(
        labels, 
        padding='max_length', 
        max_length=MAX_LENGTH, 
        truncation=True, 
        return_tensors='pt'
    )['input_ids'][:, 0] # get first token of label
    
    # put label at the last index
    examples['labels'] = torch.full_like(examples['input_ids'], IGNORE_INDEX)
    examples['labels'][:, -1] = labels

    return examples

train_dataset = preprocess(X, y)

# ===== Cell 35 =====
# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]=TRAIN_DIR

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="false"

def accuracy_metric(x):
    labels = x.label_ids[:, -1]
    # predictions = x.predictions[0].argmax(axis=-1)[:, -2]  # uncomment for gpt-neox
    predictions = x.predictions.argmax(axis=-1)[:, -2]
    return {
        'accuracy': accuracy_score(y_true=labels, y_pred=predictions),
    }

train_ds = Dataset.from_dict(train_dataset)

batch_size = 8

training_args = TrainingArguments(
    output_dir=TRAIN_DIR,
    # overwrite_output_dir=True,
    eval_strategy="epoch",
    # use a smaller learning rate and fewer epochs since the dataset is small and we are fine-tuning a pre-trained model
    learning_rate=1e-05,
    num_train_epochs=5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    report_to="wandb", # optional, remove if you don't want to log to wandb
    # use_cpu=True, # uncomment if you want to train on CPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=train_ds,
    compute_metrics=accuracy_metric
)

# ===== Cell 37 =====
import wandb 

_ = trainer.train()

# ===== Cell 38 =====
trainer.save_model(TRAIN_DIR)

# ===== Cell 40 =====
tokenizer.pad_token = tokenizer.eos_token
test_examples = mqlni_model.generate_factual_dataset(100, sampler=mqlni_model.sample_input_tree_balanced, return_tensors=False)
X = [example['input_ids'] for example in test_examples]
y = [example['labels'] for example in test_examples]
test_dataset = preprocess(X, y)
test_ds = Dataset.from_dict(test_dataset)

# ===== Cell 41 =====
results = trainer.evaluate(train_ds)
print(f"Train set evaluation results: {results}")

# ===== Cell 42 =====
results = trainer.evaluate(test_ds)
print(f"Test set evaluation results: {results}")

# ===== Cell 43 =====
wandb.finish()
