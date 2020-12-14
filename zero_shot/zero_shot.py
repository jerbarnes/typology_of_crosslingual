import numpy as np
import pandas as pd
import os
import tensorflow as tf
import functools
from IPython.utils.text import columnize
from tqdm.notebook import tqdm
from data_preparation import data_preparation_pos, data_preparation_sentiment
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import sys
sys.path.append("..")
from utils import utils, model_utils, pos_utils
from fine_tuning import fine_tuning

def is_tested(lang, results_path):
    if os.path.isfile(results_path):
        results = pd.read_excel(results_path, sheet_name=None)
        # Using accuracy sheet because it it common to both tasks
        return lang in results["Accuracy"].columns
    else:
        return False

def sort_langs_by_status(model_name, task, data_path, checkpoints_path, all_langs, results_path, excluded=[]):
    """Return lists of languages left to test, already tested languages, languages that have not been trained and
    languages that cannot be trained."""
    trained_langs, cannot_train_langs, untrained_langs = fine_tuning.sort_langs_by_status(
        model_name, task, data_path, checkpoints_path, all_langs, excluded=[]
    )

    remaining_langs = []
    tested_langs = []
    for lang in trained_langs:
        if is_tested(lang, results_path):
            tested_langs.append(lang)
        else:
            remaining_langs.append(lang)

    return remaining_langs, tested_langs, untrained_langs, cannot_train_langs

def get_global_testing_state(data_path, short_model_name, experiment, checkpoints_path, results_path):
    """Print the testing state of an experiment (languages tested, not yet tested, not yet trained, cannot be trained)
    and return the next language to be tested."""
    # Infer task from data path
    if "sentiment" in data_path:
        task = "sentiment"
    else:
        task = "pos"

    # Get full model names
    model_name, full_model_name = model_utils.get_full_model_names(short_model_name)
    # Get all languages that belong to the experiment
    all_langs = utils.get_langs(experiment)

    # Sort languages according to their status
    if task == "sentiment" and experiment == "tfm":
        excluded = ["Turkish", "Japanese", "Russian"]
    else:
        excluded = []
    remaining_langs, tested_langs, untrained_langs, cannot_train_langs = sort_langs_by_status(
        model_name, task, data_path, checkpoints_path, all_langs, results_path, excluded
    )

    # Print status
    if remaining_langs:
        training_lang = remaining_langs[0]
        print("{:<20}".format("Evaluating with:      "), training_lang, "\n")
        training_lang = utils.name_to_code[training_lang]
        print(columnize(["Already evaluated:   "] + tested_langs, displaywidth=150))
        print(columnize(["Not yet evaluated:   "] + remaining_langs[1:], displaywidth=150))
        print(columnize(["Not yet trained:     "] + untrained_langs, displaywidth=150))
        print(columnize(["Cannot train:        "] + cannot_train_langs, displaywidth=150))
    else:
        print("No languages remaining", "\n")
        print(columnize(["Already evaluated:   "] + tested_langs, displaywidth=150))
        print(columnize(["Still to train:      "] + untrained_langs, displaywidth=150))
        print(columnize(["Cannot train:        "] + cannot_train_langs, displaywidth=150))
        training_lang = None
        if input("Retrain language? ") == "y":
            while training_lang not in all_langs:
                training_lang = input("Language to re-train: ")
            training_lang = utils.name_to_code[training_lang]

    return training_lang

class Tester:
    """Class designed to carry out zero-shot evaluation."""
    def __init__(self, data_path, results_path, short_model_name, task, checkpoint_dir,
                 experiment, max_length, batch_size, num_labels, tagset=None):
        """
        Parameters:
        data_path: Path to all data for the task.
        results_path: Path to the results xlsx file.
        short_model_name: 'mbert' or 'xlm-roberta'.
        task: 'pos' or 'sentiment'.
        checkpoint_dir: Path to all model checkpoints.
        experiment: 'acl' or 'tfm'.
        max_length: Maximum example length in tokens (should be the same used in training).
        batch_size: Batch size for predicting.
        num_labels: Number of target labels.
        tagset: List of all possible tags (PoS only).
        """
        self.data_path = data_path
        self.results_path = results_path
        self.task = task
        self.checkpoint_dir = checkpoint_dir
        self.experiment = experiment
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_labels = num_labels
        if tagset:
            self.tagset = tagset
            self.label_map = {label: i for i, label in enumerate(tagset)}
        self.current_model_lang = None # Stands for the language the current weights correspond to

        # Included languages
        self.included_langs = utils.get_langs(experiment)
        if self.experiment == "tfm" and self.task == "sentiment":
            self.included_langs = [lang for lang in self.included_langs if (
                lang not in ["Russian", "Turkish", "Japanese"]
            )]

        # Model
        self.short_model_name = short_model_name
        self.model_name, self.full_model_name = model_utils.get_full_model_names(short_model_name)
        self.model = None # Do not create it yet
        self.tokenizer = None

    def setup_model(self, gpu_growth=True):
        """Build model and its corresponding tokenizer."""
        if gpu_growth:
            model_utils.set_tf_memory_growth()
        self.model, self.tokenizer = model_utils.create_model(self.short_model_name,
                                                              self.task,
                                                              self.num_labels)

    def check_if_model_exists(self):
        """Assert the model has been created before executing a function."""
        assert self.model and self.tokenizer, "You must first create the model with 'setup_model' method."

    def handle_oom(self, f, *args, **kwargs):
        """
        Handles tf.errors.ResourceExhaustedError when running a function, forcing it to retry.
        Extra arguments and keyword arguments are passed to the function.
        """
        while True:
            try:
                output = f(*args, **kwargs)
            except tf.errors.ResourceExhaustedError:
                print("\nOut of memory, retrying...")
                continue
            break
        return output

    def load_data(self, lang_path):
        """
        Load test data for the language whose path is provided.

        Parameters:
        lang_path: Path to the language's data.

        Returns:
        Data in its original format (pd.DataFrame or list of dicts).
        Data in tf.data.Dataset format.
        Number of batches in the TF dataset.
        """
        if self.task == "pos":
            data, dataset = data_preparation_pos.load_dataset(
                lang_path, self.tokenizer, self.max_length, self.short_model_name,
                tagset=self.tagset, dataset_name="test"
            )
        elif self.task == "sentiment":
            data, dataset = data_preparation_sentiment.load_dataset(
                lang_path, self.tokenizer, self.max_length, self.short_model_name,
                balanced=False, limit=None, dataset_name="test"
            )

        dataset, batches = model_utils.make_batches(
            dataset, self.batch_size, repetitions=1, shuffle=False
        )

        return data, dataset, batches

    def get_scores_pos(self, preds, data):
        """Calculate accuracy for the given PoS predictions (at word level)."""
        all_words = [] # Full words for all the dataset
        all_labels = [] # Labels for all the dataset
        real_tokens = [] # Indexes of non-padding tokens
        subword_locs = [] # Start and end index for every subword
        acc_lengths = 0 # Accumulated lengths of the examples in subword tokens

        for i in range(len(data)):
            all_words.extend(data[i]["tokens"]) # Full words
            all_labels.extend([self.label_map[label] for label in data[i]["tags"]])
            _, _, idx_map = self.tokenizer.subword_tokenize(data[i]["tokens"], data[i]["tags"])

            # Examples always start at a multiple of max_length
            # Where they end depends on the number of resulting subwords
            example_start = i * self.max_length
            example_end = example_start + len(idx_map)
            real_tokens.extend(np.arange(example_start, example_end, dtype=int))

            # Get subword starts and ends
            sub_ids, sub_starts, sub_lengths = np.unique(idx_map, return_counts=True, return_index=True)
            sub_starts = sub_starts[sub_lengths > 1] + acc_lengths
            sub_ends = sub_starts + sub_lengths[sub_lengths > 1]
            subword_locs.extend(np.array([sub_starts, sub_ends]).T.tolist())
            acc_lengths += len(idx_map)

        filtered_preds = preds[0].argmax(axis=-1).flatten()[real_tokens].tolist()
        filtered_logits = preds[0].reshape(
            (preds[0].shape[0] * preds[0].shape[1], preds[0].shape[2])
        )[real_tokens]
        new_preds = pos_utils.reconstruct_subwords(subword_locs, filtered_preds, filtered_logits)

        assert len(new_preds) == len(all_labels), "Prediction and truth lengths do not match"
        return (np.mean(np.array(new_preds) == np.array(all_labels)),)

    def get_scores_sentiment(self, preds, data):
        """Calculate accuracy and macro-averaged precision, recall and F1 score for the given sentiment
        predictions."""
        clean_preds = preds[0].argmax(axis=-1)
        y_true = data["sentiment"].values

        accuracy = accuracy_score(y_true, clean_preds)
        precision = precision_score(y_true, clean_preds, average="macro", zero_division=0)
        recall = recall_score(y_true, clean_preds, average="macro", zero_division=0)
        f1 = f1_score(y_true, clean_preds, average="macro", zero_division=0)
        return accuracy, precision, recall, f1

    def test_on_lang(self, training_lang, testing_lang):
        """
        Test a model fine-tuned on a particular language over another (or the same) language's test
        set and return the resulting scores.

        Parameters:
        training_lang: Language the desired model was fine-tuned on (ISO code or full name).
        testing_lang: Language to test the model on (ISO code or full name).

        Returns:
        Accuracy score if the task is PoS tagging.
        Accuracy, precision, recall and F1 scores (macro-averaged) if the task is sentiment analysis.
        """
        self.check_if_model_exists()
        # Here we expect ISO codes, transform if full name is given
        if testing_lang in utils.name_to_code.keys():
            testing_lang = utils.name_to_code[testing_lang]
        if training_lang in utils.name_to_code.keys():
            training_lang = utils.name_to_code[training_lang]

        # Check if the correct weights are loaded
        if self.current_model_lang != training_lang:
            self.set_model_lang(training_lang)

        lang_path = os.path.join(self.data_path, testing_lang)
        data, dataset, batches = self.load_data(lang_path)
        preds = self.handle_oom(self.model.predict, dataset, steps=batches, verbose=1)
        scores = (self.get_scores_pos if self.task == "pos" else self.get_scores_sentiment)(preds, data)
        return scores

    def set_model_lang(self, training_lang):
        """Load the model weights that correspond to a fine-tuning language."""
        self.check_if_model_exists()
        weights_path = self.checkpoint_dir + training_lang + "/"
        weights_filename = "{}_{}.hdf5".format(self.model_name, self.task)
        self.model.load_weights(weights_path + weights_filename)
        print("Using weights from", weights_path + weights_filename)
        self.current_model_lang = training_lang

    def build_results_table(self, results):
        """Transforms results into a table."""
        results = np.array(results, dtype=object)
        columns = ["Language", "Accuracy"]
        if self.task == "sentiment":
            columns.extend(["Macro_Precision", "Macro_Recall", "Macro_F1"])
        table = pd.DataFrame(results, columns=columns)
        table = utils.order_table(table, self.experiment)
        return table

    def update_results_file(self, training_lang, table):
        """Update the results file with new data from a given training language."""
        # Transform from ISO if necessary
        if training_lang in utils.code_to_name.keys():
            training_lang = utils.code_to_name[training_lang]
        # 'results' will be a dict where every key is a metric and value the results for the metric
        if os.path.isfile(self.results_path):
            results = pd.read_excel(self.results_path, sheet_name=None)
        else:
            results = dict.fromkeys(table.columns[1:].values,
                                    pd.DataFrame({"Language": table["Language"].values}))
        with pd.ExcelWriter(self.results_path) as writer:
            for sheet_name, df in results.items():
                # Add each the column for each metric (=sheet_name) in the corresponding sheet
                df[training_lang] = table[sheet_name]
                df.to_excel(writer, index=False, sheet_name=sheet_name)

    def evaluate_lang(self, training_lang, write_to_file=False):
        """Evaluate a fine-tuned model on all languages. Update results file if 'write_to_file' is set
        to True, return a pd.DataFrame otherwise."""
        self.check_if_model_exists()
        if training_lang in utils.name_to_code.keys():
            training_lang = utils.name_to_code[training_lang]

        # Load weights for this language
        self.set_model_lang(training_lang)

        # Get evaluation results
        eval_results = []
        for testing_lang in tqdm(self.included_langs, leave=False):
            scores = self.test_on_lang(training_lang, utils.name_to_code[testing_lang])
            eval_results.append((testing_lang, *scores))

        # Build table and either return it or update the file with it
        training_lang = utils.code_to_name[training_lang]
        table = self.build_results_table(eval_results)
        if write_to_file:
            print("Updating {} after evaluating {} with {}.".format(self.results_path,
                                                                    training_lang,
                                                                    self.short_model_name))
            self.update_results_file(training_lang, table)
        else:
            return table

    def batch_evaluate(self):
        self.check_if_model_exists()
        # Find languages that have not been evaluated yet
        params = {"model_name": self.model_name,
                  "task": self.task,
                  "data_path": self.data_path,
                  "checkpoints_path": self.checkpoint_dir,
                  "all_langs": self.included_langs,
                  "results_path": self.results_path}
        remaining_langs, tested_langs, _, _ = sort_langs_by_status(**params)
        print(columnize(["Already evaluated:   "] + tested_langs, displaywidth=150))

        for training_lang in tqdm(remaining_langs):
            print("Now evaluating", training_lang)
            self.evaluate_lang(training_lang, write_to_file=True)