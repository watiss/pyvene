import torch
import json
import random
from sklearn.metrics import accuracy_score

# copied as is from the MQNLI.ipynb
def create_causal_model():
    from pyvene import CausalModel

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

    variables = list(parents.keys())

    mqlni_model = CausalModel(variables, values, parents, functions, pos=pos)

    return mqlni_model

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

def preprocess(X, y, tokenizer, ignore_index=-100, max_length=64):
    tokenizer.pad_token = tokenizer.eos_token
    # Left-pad so that the real prompt always ends at the final positions. This is
    # what makes indexing the answer at [:, -1] (and its predicting logits at
    # [:, -2]) correct for every example regardless of prompt length.
    tokenizer.padding_side = 'left'

    examples = [preprocess_input(x) for x in X]
    labels = [preprocess_output(y) for y in y]

    # First token id of each (single-token) answer string.
    label_ids = tokenizer(labels, add_special_tokens=False)['input_ids']
    for ids in label_ids:
        # We score a single answer token; assert the relation is one token.
        assert len(ids) >= 1
    label_ids = [ids[0] for ids in label_ids]

    # Append the answer token to the prompt so the model is teacher-forced on it
    # and the answer ends up at the last (non-pad) position after left padding.
    prompt_ids = tokenizer(examples, add_special_tokens=False)['input_ids']
    full_ids = [p + [a] for p, a in zip(prompt_ids, label_ids)]
    # Truncate from the LEFT so the appended answer token is never dropped.
    full_ids = [ids[-max_length:] for ids in full_ids]

    # Left-pad the full (prompt + answer) sequences to MAX_LENGTH.
    examples = tokenizer.pad(
        {'input_ids': full_ids},
        padding='max_length',
        max_length=max_length,
        return_tensors='pt',
    )
    assert examples['input_ids'].shape[1] == max_length

    # Put the answer label at the last index; everything else is ignored.
    # With left padding, position -1 is the appended answer token, and the
    # logits at position -2 are the ones that predict it.
    examples['labels'] = torch.full_like(examples['input_ids'], ignore_index)
    examples['labels'][:, -1] = torch.tensor(label_ids)

    return examples

def preprocess_counterfactual(data, tokenizer, ignore_index=-100, max_length=64):
    tokenizer.pad_token = tokenizer.eos_token
    # Left-pad so the real prompt always ends at the final positions. This is what
    # makes indexing the answer at [:, -1] (and its predicting logits at [:, -2])
    # correct for every example regardless of prompt length.
    tokenizer.padding_side = 'left'

    def tokenize_with_label(prompt, answer, tokenizer, max_length):
        """Tokenize `prompt`, append the single answer token (teacher forcing), and
        left-pad/-truncate to max_length. Returns the padded inputs plus the answer
        token id so the caller can place the label at the last index.
        """
        # First token id of the (single-token) answer string.
        answer_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
        # We score a single answer token; assert the relation is one token.
        assert len(answer_ids) >= 1
        answer_id = answer_ids[0]

        # Append the answer token to the prompt so it ends up at the last
        # (non-pad) position after left padding.
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        full_ids = prompt_ids + [answer_id]
        # Truncate from the LEFT so the appended answer token is never dropped.
        full_ids = full_ids[-max_length:]

        # Left-pad the full (prompt + answer) sequence to max_length.
        inputs = tokenizer.pad(
            {'input_ids': [full_ids]},
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
        )
        assert inputs['input_ids'].shape[1] == max_length
        return inputs, answer_id

    preprocessed_data = []
    for d in data:
        base = preprocess_input(d['input_ids'])
        sources = [preprocess_input(d['source_input_ids'][0])]
        label = preprocess_output(d['labels'])
        base_label = preprocess_output(d['base_labels'])

        preprocessed = {}
        # Base input is teacher-forced on the BASE label so its last position
        # holds the base answer token (matching how the model was trained).
        preprocessed['input'], _ = tokenize_with_label(base, base_label, tokenizer, max_length)
        # Sources only provide activations to read; the trailing token is
        # arbitrary, so teacher-force on the base label for consistent shape.
        src_inputs, _ = tokenize_with_label(sources[0], base_label, tokenizer, max_length)
        preprocessed['source'] = [src_inputs]

        # Place the counterfactual label at the last index; everything else is
        # ignored. With left padding, position -1 is the answer token and the
        # logits at position -2 are the ones that predict it.
        label_id = tokenizer(label, add_special_tokens=False)['input_ids']
        assert len(label_id) >= 1
        preprocessed['label'] = torch.full_like(
            preprocessed['input']['input_ids'], ignore_index)
        preprocessed['label'][:, -1] = label_id[0]

        # Repeat for the base label.
        base_label_id = tokenizer(base_label, add_special_tokens=False)['input_ids']
        assert len(base_label_id) >= 1
        preprocessed['base_label'] = torch.full_like(
            preprocessed['input']['input_ids'], ignore_index)
        preprocessed['base_label'][:, -1] = base_label_id[0]

        preprocessed['intervention_id'] = torch.tensor(d['intervention_id'])
        preprocessed_data.append(preprocessed)

    return preprocessed_data

def compute_metrics(eval_preds, eval_labels):
    accuracy = accuracy_score(
        y_pred=eval_preds[..., -2].squeeze().clone().detach().cpu().numpy(), 
        y_true=eval_labels[..., -1].squeeze().clone().detach().cpu().numpy()
    )
    return {
        "accuracy": accuracy
    }

def create_counterfactual_dataset(
    mqlni_model,
    size,
    # specifies how many inputs we want per intervention that is sampled
    batch_size,
):
    def sample_intervention(*args, **kwargs):
        return {
            'NegP' : random.choice(mqlni_model.values['NegP'])
        }

    def intervention_id(*args, **kwargs):
        return 0

    dataset = mqlni_model.generate_counterfactual_dataset(
        size, intervention_id, batch_size, 
        sampler=mqlni_model.sample_input_tree_balanced, intervention_sampler=sample_intervention, return_tensors=False
    )

    return dataset

def accuracy_metric(x):
    # With left padding the answer token sits at position -1, so its label is at
    # -1 and the logits predicting it (position i predicts token i+1) are at -2.
    labels = x.label_ids[:, -1]
    # predictions = x.predictions[0].argmax(axis=-1)[:, -2]  # uncomment for gpt-neox
    predictions = x.predictions.argmax(axis=-1)[:, -2]
    return {
        'accuracy': accuracy_score(y_true=labels, y_pred=predictions),
    }