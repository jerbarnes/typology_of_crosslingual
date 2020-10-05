import glob
import numpy as np
import sys
sys.path.append("..")
from data_preparation.data_preparation_pos import convert_examples_to_tf_dataset, read_conll

def load_data(path, batch_size, tokenizer, tagset, max_length):
    """Loads conllu file, returns a list of dictionaries (one for each sentence) and a TF dataset"""
    test_data = read_conll(glob.glob(path + "/*-test.conllu")[0])
    test_examples = [{"id": sent_id, "tokens": tokens, "tags": tags} for sent_id, tokens, tags in zip(test_data[0], 
                                                                                                      test_data[1],
                                                                                                      test_data[2])]
    # In case some example is over max length
    test_examples = [example for example in test_examples if len(tokenizer.subword_tokenize(example["tokens"], 
                                                                                            example["tags"])[0]) <= max_length]
    test_dataset = convert_examples_to_tf_dataset(examples=test_examples, tokenizer=tokenizer, tagset=tagset, max_length=256)
    test_dataset = test_dataset.batch(batch_size)
    return test_examples, test_dataset

def filter_padding_tokens(test_examples, preds, label_map, tokenizer):
    """Filters padding tokens, labels, predictions and logits, then returns these as flattened lists"""
    filtered_preds = []
    labels = []
    tokens = []
    logits = []

    for i in range(len(test_examples)):
        example_tokens, example_labels, _ = tokenizer.subword_tokenize(test_examples[i]["tokens"], test_examples[i]["tags"])
        example_labels = [label_map[label] for label in example_labels]
        example_preds = preds[0].argmax(axis=-1)[i, :len(example_labels)]
        example_logits = preds[0][i, :len(example_labels)]
        filtered_preds.extend(example_preds)
        labels.extend(example_labels)
        tokens.extend(example_tokens)
        logits.extend(example_logits)
        
    return tokens, labels, filtered_preds, logits

def find_subword_locations(tokens):
    """Finds the starting and ending index of words that have been broken into subwords"""
    subword_locations = []

    for i in range(len(tokens)):
        if tokens[i].startswith("##") and not(tokens[i-1].startswith("##")):
            start = i - 1
        if not(tokens[i].startswith("##")) and tokens[i-1].startswith("##") and i != 0:
            end = i
            subword_locations.append((start, end))
            
    return subword_locations

def reconstruct_subwords(subword_locations, tokens, labels, filtered_preds, logits):
    """Assemble subwords back into the original word in the global lists of tokens, labels and predictions,
    and select a predicted tag"""
    new_tokens = []
    new_preds = []
    new_labels = []
    prev_end = 0

    for start, end in subword_locations:
        if len(set(filtered_preds[start:end])) > 1:
            # Subword predictions do not all agree
            temp = np.array([(M.max(), M.argmax()) for M in logits[start:end]])
            prediction = temp[temp[:,0].argmax(), 1]
        else:
            prediction = filtered_preds[start]
        new_preds += filtered_preds[prev_end:start] + [prediction]
        token = "".join(tokens[start:end]).replace("##", "")
        new_tokens += tokens[prev_end:start] + [token]
        new_labels += labels[prev_end:start] + [labels[start]]
        prev_end = end

    # Last subword onwards
    new_preds += filtered_preds[prev_end:]
    new_tokens += tokens[prev_end:]
    new_labels += labels[prev_end:]
    
    return new_tokens, new_labels, new_preds