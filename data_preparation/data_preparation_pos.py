from transformers import BertTokenizer, XLMRobertaTokenizer
from transformers.data.processors.utils import InputFeatures
import tensorflow as tf
import logging
import glob
import numpy as np

def read_conll(input_file):
        """Reads a conllu file."""
        ids = []
        texts = []
        tags = []
        #
        text = []
        tag = []
        idx = None
        for line in open(input_file, encoding="utf-8"):
            if line.startswith("# sent_id ="):
                idx = line.strip().split()[-1]
                ids.append(idx)
            elif line.startswith("#"):
                pass
            elif line.strip() == "":
                texts.append(text)
                tags.append(tag)
                text, tag = [], []
            else:
                try:
                    splits = line.strip().split("\t")
                    token = splits[1] # the token
                    label = splits[3] # the UD POS Tag label
                    text.append(token)
                    tag.append(label)
                except:
                    print(line)
                    print(idx)
                    raise
        return ids, texts, tags

class MBERT_Tokenizer(BertTokenizer):
    """M-BERT tokenizer adapted for PoS tagging."""
    def subword_tokenize(self, tokens, labels):
        """
        Propagate tags through subwords.

        Parameters:
        tokens: List of word tokens.
        labels: List of PoS tags.

        Returns:
        List of subword tokens.
        List of propagated tags.
        List of indexes mapping subwords to the original word.
        """
        # This propogates the label over any subwords that
        # are created by the byte-pair tokenization for training

        # IMPORTANT: For testing, you will have to undo this step by combining
        # the subword elements, and

        split_tokens, split_labels = [], []
        idx_map = []
        for ix, token in enumerate(tokens):
            sub_tokens = self.wordpiece_tokenizer.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                split_labels.append(labels[ix])
                idx_map.append(ix)
        return split_tokens, split_labels, idx_map

class XLMR_Tokenizer(XLMRobertaTokenizer):
    """XLM-Roberta tokenizer adapted for PoS tagging."""
    def subword_tokenize(self, tokens, labels):
        """
        Propagate tags through subwords.

        Parameters:
        tokens: List of word tokens.
        labels: List of PoS tags.

        Returns:
        List of subword tokens.
        List of propagated tags.
        List of indexes mapping subwords to the original word.
        """
        # This propogates the label over any subwords that
        # are created by the byte-pair tokenization for training

        # IMPORTANT: For testing, you will have to undo this step by combining
        # the subword elements, and

        split_tokens, split_labels = [], []
        idx_map = []
        for ix, token in enumerate(tokens):
            sub_tokens = self.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                split_labels.append(labels[ix])
                idx_map.append(ix)
        return split_tokens, split_labels, idx_map

def bert_convert_examples_to_tf_dataset(examples, tokenizer, tagset, max_length):
    """Return a TF dataset adapted for M-BERT."""
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        tokens = e["tokens"]
        labels = e["tags"]
        label_map = {label: i for i, label in enumerate(tagset)} # Tags to indexes

        # Tokenize subwords and propagate labels
        split_tokens, split_labels, idx_map = tokenizer.subword_tokenize(tokens, labels)

        # Create features
        input_ids = tokenizer.convert_tokens_to_ids(split_tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * max_length
        label_ids = [label_map[label] for label in split_labels]

        padding = [0] * (max_length - len(input_ids))
        input_ids += padding
        attention_mask += padding
        label_ids += padding

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label_ids
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([None]),
        ),
    )

def roberta_convert_examples_to_tf_dataset(examples, tokenizer, tagset, max_length):
    """Return a TF dataset adapted for XLM-Roberta."""
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        tokens = e["tokens"]
        labels = e["tags"]
        label_map = {label: i for i, label in enumerate(tagset)} # Tags to indexes

        # Tokenize subwords and propagate labels
        split_tokens, split_labels, idx_map = tokenizer.subword_tokenize(tokens, labels)

        # Create features
        input_ids = tokenizer.convert_tokens_to_ids(split_tokens)
        attention_mask = [1] * len(input_ids)
        label_ids = [label_map[label] for label in split_labels]

        padding = [0] * (max_length - len(input_ids))
        input_ids += padding
        attention_mask += padding
        label_ids += padding

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label=label_ids
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
            },
            tf.TensorShape([None]),
        ),
    )

def load_dataset(lang_path, tokenizer, max_length, short_model_name, tagset,
                 dataset_name="test", sample=None, sample_idxs=None):
    """Load conllu file, return a list of dictionaries (one for each sentence) and a TF dataset. Use sample to
    get a subset of the given size."""
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) # Avoid max length warning
    convert_functions = {"mbert": bert_convert_examples_to_tf_dataset,
                         "xlm-roberta": roberta_convert_examples_to_tf_dataset}
    data = read_conll(glob.glob(lang_path + "/*{}.conllu".format(dataset_name.split("_")[0]))[0])
    examples = [{"id": sent_id, "tokens": tokens, "tags": tags} for sent_id, tokens, tags in zip(data[0],
                                                                                                 data[1],
                                                                                                 data[2])]
    # In case some example is over max length
    examples = [example for example in examples if len(tokenizer.subword_tokenize(example["tokens"],
                                                                                  example["tags"])[0]) <= max_length]
    if sample:
        examples = np.random.choice(examples, size=sample)
    elif sample_idxs is not None and dataset_name.startswith("train"):
        examples = np.array(examples)[sample_idxs].tolist()

    dataset = convert_functions[short_model_name](examples=examples, tokenizer=tokenizer,
                                                  tagset=tagset, max_length=max_length)
    return examples, dataset
    # This loops 3 times over the same data, including the convert to TF, could it be done in one?
