
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from pyvene import (
    IntervenableModel,
    RotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig
)
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from collections import Counter
from pyvene.models.gpt2.modelings_intervenable_gpt2 import create_gpt2_lm
from MQNLI_utils import preprocess_counterfactual, create_causal_model, create_counterfactual_dataset, compute_metrics

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

# DAS_SUBSPACE_PARTITION = 4
# DAS_SUBSPACE_PARTITION = 8
# DAS_SUBSPACE_PARTITION = 16
# DAS_SUBSPACE_PARTITION = 32
# DAS_SUBSPACE_PARTITION = 64
DAS_SUBSPACE_PARTITION = 128
DAS_LAYER = 10

TRAIN_DIR = 'mqnli_factual'
DAS_DIR = 'mqnli_das_{}_{}'.format(DAS_LAYER, DAS_SUBSPACE_PARTITION)

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

################ CREATE INTERVENABLE MODEL ###############

_, tokenizer, model = create_gpt2_lm(name=TRAIN_DIR)

# Here we set up our alignment. We will search for a subspace with a dimension of 128 (a fourth of our model's
# hidden dimension size) in the residual stream of the 10th layer of the model (out of 12 overall transformer
# layers).
config = IntervenableConfig(
    model_type=type(model),
    representations=[
        RepresentationConfig(
            DAS_LAYER,  # layer
            "block_output",  # intervention type
            "pos",  # intervention unit is now aligned with tokens
            1,  # max number of unit
            subspace_partition=[[0, DAS_SUBSPACE_PARTITION]], 
            # intervention_link_key=0,
        )
    ],
    intervention_types=RotatedSpaceIntervention,
)

intervenable = IntervenableModel(config, model, use_fast=True)
intervenable.set_device('cuda')
intervenable.disable_model_gradients()

##################################################

########## CREATE COUNTERFACTUAL DATASET ##########

# our interpretability experiments rely on a counterfactual dataset: for a given input, we want to consider what
# might happen had the value of `NegP` changed while all else remained the same. Fortunately, this is precisely
# what our causal model can provide us with! We can generate a counterfactual dataset by sampling inputs that
# only vary from each other on the `NegP` node.

mqlni_model = create_causal_model()

dataset = create_counterfactual_dataset(mqlni_model, 100, 2)

print(dataset[0]['base_labels']['QP_S'])
print(dataset[0]['labels']['QP_S'])
# check that base labels are diverse (should be guaranteed by sampling from balanced input tree)
print(Counter([d['base_labels']['QP_S'] for d in dataset]))
# check that counterfactuals labels are diverse (note that this could be skewed by the node's effect on the final output)
print(Counter([d['labels']['QP_S'] for d in dataset]))

##################################################

########## CREATE DATALODER AND OPTIMIZER ##########

train_dataset = preprocess_counterfactual(dataset, tokenizer)
dataloader = DataLoader(train_dataset, batch_size=2)

optimizer_params = []
for k, v in intervenable.interventions.items():
    optimizer_params += [{"params": v.rotate_layer.parameters()}]
    break
optimizer = torch.optim.Adam(optimizer_params, lr=0.001)

##################################################

#################### TRAIN #######################

def compute_loss(outputs, labels):
    # Shift so that tokens < n predict n
    shift_logits = outputs[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

intervenable.model.train()  # train enables drop-off but no grads
print("intervention trainable parameters: ", intervenable.count_parameters())

epochs = 3
train_iterator = trange(0, int(epochs), desc="Epoch")
batch_size = 8
gradient_accumulation_steps = 1
MAX_LENGTH = 64

total_step = 0
for epoch in train_iterator:
    epoch_iterator = tqdm(
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True
        ),
        desc=f"Epoch: {epoch}",
        position=0,
        leave=True
    )
    for batch in epoch_iterator:
        inputs = {k: v.to(intervenable.get_device()) for k, v in batch['input'].items()}
        sources = [{k: v.to(intervenable.get_device()) for k, v in s.items()} for s in batch['source']]
        _, counterfactual_outputs = intervenable(
            inputs,
            sources,
            {"sources->base": ([[[MAX_LENGTH - 2]] * batch_size], [[[MAX_LENGTH - 2]] * batch_size])},
            subspaces=[[[0]] * batch_size],
        )

        eval_metrics = compute_metrics(
            counterfactual_outputs.logits.argmax(-1), batch["label"].to(intervenable.get_device())
        )

        # loss and backprop
        loss = compute_loss(
            counterfactual_outputs.logits, batch["label"].to(intervenable.get_device())
        )

        epoch_iterator.set_postfix({"loss": loss.item(), "acc": eval_metrics["accuracy"]})

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        if total_step % gradient_accumulation_steps == 0:
            optimizer.step()
            intervenable.set_zero_grad()
        total_step += 1

##################################################

intervenable.save(DAS_DIR)
