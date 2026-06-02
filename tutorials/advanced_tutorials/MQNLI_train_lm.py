
import os
import random
import numpy as np
import torch
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score
from pyvene.models.gpt2.modelings_intervenable_gpt2 import create_gpt2_lm
from MQNLI_utils import print_premise, print_hypothesis, preprocess, create_causal_model, accuracy_metric

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

TRAIN_DIR = 'mqnli_factual'

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

mqlni_model = create_causal_model()

examples = mqlni_model.generate_factual_dataset(
    500,
    sampler=mqlni_model.sample_input_tree_balanced,
    return_tensors=False,
)

random.shuffle(examples)

# Three-way split: train / validation / test.
# - train: used to update model weights.
# - validation: used for per-epoch eval + best-model selection (held out, so it
#   actually guards against a bad optimizer step degrading the model).
# - test: untouched during training, used only for final reporting below.
n_train = 350
n_val = 50
train_examples = examples[:n_train]
val_examples = examples[n_train:n_train + n_val]
test_examples = examples[n_train + n_val:]
assert len(train_examples) + len(val_examples) + len(test_examples) == len(examples)

X = [example['input_ids'] for example in train_examples]
y = [example['labels'] for example in train_examples]

X_val = [example['input_ids'] for example in val_examples]
y_val = [example['labels'] for example in val_examples]

print_premise(X[0])
print_hypothesis(X[0])
print(y[0]['QP_S'])

config, tokenizer, model = create_gpt2_lm()

train_dataset = preprocess(X, y, tokenizer)
val_dataset = preprocess(X_val, y_val, tokenizer)

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]=TRAIN_DIR

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="false"

train_ds = Dataset.from_dict(train_dataset)
val_ds = Dataset.from_dict(val_dataset)

batch_size = 8

training_args = TrainingArguments(
    output_dir=TRAIN_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    ######
    learning_rate=1e-4,
    num_train_epochs=40,
    ######
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    report_to="wandb", # optional, remove if you don't want to log to wandb
    # use_cpu=True, # uncomment if you want to train on CPU for some reason
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,  # held-out validation for best-model selection
    compute_metrics=accuracy_metric
)

import wandb 

_ = trainer.train()

trainer.save_model(TRAIN_DIR)

X = [example['input_ids'] for example in test_examples]
y = [example['labels'] for example in test_examples]
test_dataset = preprocess(X, y, tokenizer)
test_ds = Dataset.from_dict(test_dataset)

results = trainer.evaluate(train_ds)
print(f"Train set evaluation results: {results}")

results = trainer.evaluate(val_ds)
print(f"Validation set evaluation results: {results}")

results = trainer.evaluate(test_ds)
print(f"Test set evaluation results: {results}")

wandb.finish()
