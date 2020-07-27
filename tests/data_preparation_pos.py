from transformers import BertTokenizer
from transformers.data.processors.utils import InputFeatures
import tensorflow as tf

class ABSATokenizer(BertTokenizer):
    def subword_tokenize(self, tokens, labels):
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
    
def convert_examples_to_tf_dataset(examples, tokenizer, tagset, max_length):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        tokens = [token["form"] for token in e] # Obtain list of tokens
        labels = [token["upos"] for token in e] # Obtain list of labels
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