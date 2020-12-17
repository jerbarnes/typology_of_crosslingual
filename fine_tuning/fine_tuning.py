import glob
from IPython.utils.text import columnize
from tqdm.notebook import tqdm
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow as tf
import itertools
from datetime import timedelta
from sklearn.metrics import classification_report, f1_score
from IPython.display import clear_output
sys.path.append("..")
import utils.utils as utils
import utils.pos_utils as pos_utils
import utils.model_utils as model_utils
import data_preparation.data_preparation_pos as data_preparation_pos
import data_preparation.data_preparation_sentiment as data_preparation_sentiment

metric_names = {"pos": "Accuracy", "sentiment": "Macro F1"}

def is_trainable(lang, data_path, task):
    """Return True if the given language has training and validation data for the given task."""
    extension = {"pos": "conllu", "sentiment": "csv"}

    for dataset in ["train", "dev"]:
        if not glob.glob(data_path + utils.name_to_code[lang] + "/*{}.{}".format(dataset, extension[task])):
            return False
    return True

def is_trained(lang, model_name, task, checkpoints_path):
    """Return True if the given language has been trained in the given task."""
    if glob.glob(checkpoints_path + "{}/{}_{}.hdf5".format(utils.name_to_code[lang], model_name, task)):
        return True
    else:
        return False

def sort_langs_by_status(model_name, task, data_path, checkpoints_path, all_langs, excluded=[]):
    """Return lists of trained languages, languages that cannot be trained and languages that have not been trained."""
    trained_langs = []
    cannot_train_langs = []
    remaining_langs = []
    for lang in all_langs:
        if lang in excluded:
            continue
        # Check if there are train and dev sets available
        elif is_trainable(lang, data_path, task):
            if is_trained(lang, model_name, task, checkpoints_path):
                trained_langs.append(lang)
            else:
                remaining_langs.append(lang)
        else:
            cannot_train_langs.append(lang)

    return trained_langs, cannot_train_langs, remaining_langs

def get_global_training_state(data_path, short_model_name, experiment, checkpoints_path):
    """Print the training state of an experiment (languages trained, not yet trained, cannot be trained)
    and return the next language to be trained."""
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
    trained_langs, cannot_train_langs, remaining_langs = sort_langs_by_status(
        model_name, task, data_path, checkpoints_path, all_langs, excluded
    )


    # Print status
    if remaining_langs:
        training_lang = remaining_langs[0]
        print("{:<20}".format("Training language:"), training_lang, "\n")
        training_lang = utils.name_to_code[training_lang]
        print(columnize(["Already trained:   "] + trained_langs, displaywidth=150))
        print(columnize(["Not yet trained:   "] + remaining_langs[1:], displaywidth=150))
        print(columnize(["Cannot train:      "] + cannot_train_langs, displaywidth=150))
    else:
        print("No languages remaining", "\n")
        print(columnize(["Already trained:   "] + trained_langs, displaywidth=150))
        print(columnize(["Cannot train:      "] + cannot_train_langs, displaywidth=150))
        training_lang = None
        if input("Retrain language? ") == "y":
            while training_lang not in all_langs:
                training_lang = input("Language to re-train: ")
            training_lang = utils.name_to_code[training_lang]

    return training_lang

def get_global_experiment_state(experiment, general_data_path, checkpoints_path, print_state=True, return_state=False):
    """
    Get overall training state of the experiment.

    Parameters:
    experiment: 'acl' or 'tfm'.
    general_data_path: Path to data directory.
    checkpoints_path: Path to where model weights are stored.
    print_state: If True, print a summary of the training state.
    return_state: If True, return a table with the experiment state.

    Returns:
    pd.DataFrame if return_state is set to True.
    """
    assert experiment in ["tfm", "acl"], "Only possible experiments are 'tfm' and 'acl'"
    langs = utils.get_langs(experiment)
    state = {}
    data_paths = {"pos": general_data_path + "ud/", "sentiment": general_data_path + "sentiment/"}
    models = ["mbert", "xlm-roberta"]
    models_print = ["M-BERT", "XLM-Roberta"]
    num_trained = {}
    num_trainable = {}

    if print_state:
        # Print table headers
        tab = "{:<15}".format("")
        header = tab + ("|{:^19}" * (len(models) + 1)).format("Trainable", *models_print) + "|"
        print(header)
        subheader = tab + "|{:^9}{:^10}".format("PoS", "Sentiment") * (len(models) + 1) + "|"
        print(subheader)
        print("{:<15}{:-^61}".format("", ""))

    for lang in langs:
        state[lang] = {}
        for task in ["pos", "sentiment"]:
            data_path = data_paths[task]
            trainable = is_trainable(lang, data_path, task)
            state[lang]["trainable_" + task] = trainable
            if trainable:
                num_trainable[task] = num_trainable.get(task, 0) + 1
            for model in models:
                model_name, full_model_name = model_utils.get_full_model_names(model)
                trained = is_trained(lang, model_name, task, checkpoints_path)
                key = "{}_{}".format(model, task)
                state[lang][key] = trained
                if trained:
                    num_trained[key] = num_trained.get(key, 0) + 1
        if print_state:
            lang_state = ["x" if state[lang]["_".join(x)] else "" for x in itertools.product(
                ["trainable"] + models, ["pos", "sentiment"]
            )]
            print("{:<15}".format(lang) + ("|{:^9}{:^10}" * (len(models) + 1)).format(*lang_state) + "|")
    if print_state:
        print("\n")
        for k, v in num_trained.items():
            model, task = k.split("_")
            print("- \033[1m{}\033[0m with model \033[1m{}\033[0m is \033[1m{:.1f}%\033[0m done".format(task.capitalize(),
                                                              models_print[models.index(model)],
                                                              v / num_trainable[task] * 100))
        progress = sum(num_trained.values()) / (len(models) * sum(num_trainable.values())) * 100
        print("\n\033[1m{}\033[0m experiment is \033[1m{:.1f}%\033[0m done".format(experiment.upper(), progress))

    if return_state:
        return pd.DataFrame(state).T.astype(int)

def get_fine_tune_scores(task, checkpoint_dir):
    param_files = glob.glob("{}*/logs/*{}_params.xlsx".format(checkpoint_dir, task))
    scores = []
    for file in param_files:
        df = pd.read_excel(file)
        model = re.search(r"([^\\/]*)_{}_params.xlsx".format(task), file).group(1)
        train_score = df.loc[df["Variable"] == "train_score", "Value"].astype(float).values[0]
        dev_score = df.loc[df["Variable"] == "dev_score", "Value"].astype(float).values[0]
        lang = utils.code_to_name[re.split(r"[\\/]", file)[-3]]
        scores.append((lang, model, train_score, dev_score))
    return pd.DataFrame(scores, columns=["Language", "Model", "Train_Score", "Dev_score"])

class Trainer:
    """General class to control the fine-tuning process of a model."""
    def __init__(self, training_lang, data_path, task, short_model_name, use_class_weights=False):
        score_functions = {"pos": self.get_score_pos, "sentiment": self.get_score_sentiment}

        self.training_lang = training_lang
        self.data_path = data_path
        self.lang_path = data_path + training_lang + "/"
        self.task = task
        if self.task == "pos":
            self.eval_info = {}
        self.metric = score_functions[task]
        self.use_class_weights = use_class_weights
        self.class_weights = None

        # Model names
        self.short_model_name = short_model_name
        self.model_name, self.full_model_name = model_utils.get_full_model_names(short_model_name)

    def build_model(self, max_length, train_batch_size, learning_rate, epochs, num_labels,
                    tagset=None, gpu_growth=True, eval_batch_size=32):
        """Create and compile model, along with its tokenizer."""
        if gpu_growth:
            model_utils.set_tf_memory_growth()
        self.model, self.tokenizer = model_utils.create_model(self.short_model_name,
                                                              self.task,
                                                              num_labels)
        self.model = model_utils.compile_model(self.model, self.task, learning_rate)
        print("Successfully built", self.model_name)
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_labels = num_labels
        if tagset:
            self.tagset = tagset
            self.label_map = {label: i for i, label in enumerate(tagset)}
        self.eval_batch_size = eval_batch_size

    def setup_checkpoint(self, checkpoints_path):
        """Setup a checkpoint directory and all necessary file names."""
        self.checkpoint_dir = checkpoints_path + self.training_lang + "/"
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if self.task == "sentiment" and self.use_class_weights:
            suffix = "_classweights"
        elif self.task == "sentiment" and not self.use_class_weights:
            suffix = "_balancedclasses"
        else:
            suffix = ""
        self.suffix = suffix
        self.checkpoint_filepath = self.checkpoint_dir + self.model_name + "_{}_checkpoint{}.hdf5".format(self.task, suffix)
        print("Checkpoint file:", self.checkpoint_filepath)
        self.temp_weights_filepath = self.checkpoint_dir + self.model_name + "_temp.hdf5"
        print("Temp weights file:", self.temp_weights_filepath)

    def setup_eval_pos(self, data, dataset_name):
        """Prepare evaluation for given PoS data."""
        self.eval_info[dataset_name] = {}
        self.eval_info[dataset_name]["all_words"] = [] # Full words for all the dataset
        self.eval_info[dataset_name]["all_labels"] = [] # Labels for all the dataset
        self.eval_info[dataset_name]["real_tokens"] = [] # Indexes of non-padding tokens
        self.eval_info[dataset_name]["subword_locs"] = [] # Start and end index for every subword
        acc_lengths = 0 # Accumulated lengths of the examples in subword tokens

        for i in range(len(data)):
            self.eval_info[dataset_name]["all_words"].extend(data[i]["tokens"]) # Full words
            self.eval_info[dataset_name]["all_labels"].extend([self.label_map[label] for label in data[i]["tags"]])
            _, _, idx_map = self.tokenizer.subword_tokenize(data[i]["tokens"], data[i]["tags"])

            # Examples always start at a multiple of max_length
            # Where they end depends on the number of resulting subwords
            example_start = i * self.max_length
            example_end = example_start + len(idx_map)
            self.eval_info[dataset_name]["real_tokens"].extend(np.arange(example_start, example_end, dtype=int))

            # Get subword starts and ends
            sub_ids, sub_starts, sub_lengths = np.unique(idx_map, return_counts=True, return_index=True)
            sub_starts = sub_starts[sub_lengths > 1] + acc_lengths
            sub_ends = sub_starts + sub_lengths[sub_lengths > 1]
            self.eval_info[dataset_name]["subword_locs"].extend(np.array([sub_starts, sub_ends]).T.tolist())
            acc_lengths += len(idx_map)

    def prepare_data(self, limit=None):
        """Load and preprocess all data."""
        datasets = {}
        dataset_names = ["train", "dev", "train_eval"]

        for dataset_name in tqdm(dataset_names):
            # Load plain data and TF dataset
            if self.task == "pos":
                data, dataset = data_preparation_pos.load_dataset(
                    self.lang_path, self.tokenizer, self.max_length, self.short_model_name,
                    tagset=self.tagset, dataset_name=dataset_name
                )
                if dataset_name != "train":
                    self.setup_eval_pos(data, dataset_name)
            elif self.task == "sentiment":
                if self.use_class_weights:
                    balanced = False
                else:
                    balanced = True
                self.balanced = balanced
                self.limit = limit
                data, dataset = data_preparation_sentiment.load_dataset(
                    self.lang_path, self.tokenizer, self.max_length, self.short_model_name,
                    balanced=balanced, limit=limit, dataset_name=dataset_name
                )
            if dataset_name == "train":
                dataset, batches = model_utils.make_batches(
                    dataset, self.train_batch_size, repetitions=self.epochs, shuffle=True
                )
            else:
                dataset, batches = model_utils.make_batches(
                    dataset, self.eval_batch_size, repetitions=1, shuffle=False
                )
            datasets[dataset_name] = (dataset, batches, data)

        self.train_dataset, self.train_batches, self.train_data = datasets["train"]
        self.dev_dataset, self.dev_batches, self.dev_data = datasets["dev"]
        self.train_eval_dataset, self.train_eval_batches, self.train_eval_data = datasets["train_eval"]

    def calc_class_weights(self):
        """
        Calculate class weights according to:

        weight(class) = total / (n_classes * n_class)

        where:
        - 'total' is the total number of examples.
        - 'n_classes' is the number of classes.
        - 'n_class' is the number of examples for the given class.

        This only applies to the sentiment task.
        """
        y = self.train_eval_data["sentiment"]
        self.class_weights = {}
        classes = np.unique(y)
        for cls in classes:
            self.class_weights[cls] = len(y) / (len(classes) * (y == cls).sum())

    def setup_training(self, load_previous_checkpoint=False, resume_from_temp=False, finish_eval=False):
        """
        Initialize the Trainer's history.

        Parameters:
        load_previous_checkpoint: If True, resume training from the last checkpoint.
        resume_from_temp: If True, resume training from the last temporal weights.
        finish_eval: Use with 'resume_from_temp' when the last epoch's evaluation could not be finished. If True,
                     temporal weights will be evaluated before resuming.
        """
        self.history = History(self)
        if resume_from_temp:
            print("Loading from", self.temp_weights_filepath)
            if os.path.isfile(self.history.log_filepath):
                self.history.load_from_last()
            self.model.load_weights(self.temp_weights_filepath)
            if finish_eval:
                print("Evaluating...")
                # Carry out the evaluation for the previous unfinished epoch
                train_preds, dev_preds = self.predict_all()
                train_score = self.metric(train_preds, self.train_eval_data, "train_eval")
                dev_score = self.metric(dev_preds, self.dev_data, "dev")

                # Update history
                if self.history.epoch_list:
                    epoch = self.history.epoch_list[-1] + 1
                else:
                    epoch = 0
                loss = np.nan # Cannot infer these, leave as missing
                epoch_duration = 0
                if dev_score > self.history.best_dev_score:
                    self.save_checkpoint(dev_score)
                    self.history.update_best_dev_score(train_score, dev_score,
                                                       epoch, epoch_duration, dev_preds)
                self.history.update_hist(epoch, loss, train_score, dev_score, epoch_duration)
        elif load_previous_checkpoint:
            print("Loading from", self.checkpoint_filepath)
            self.history.load_from_checkpoint()
            self.model.load_weights(self.checkpoint_filepath)

    def reset_to_epoch_start(self):
        """Reset the model's weights to those at the start of the epoch."""
        self.model.load_weights(self.temp_weights_filepath)

    def handle_oom(self, f, *args, **kwargs):
        """
        Handles tf.errors.ResourceExhaustedError when running a function, forcing it to retry.
        If this happened while training, it will reset the model's weights to those at the start
        of the current epoch. Extra arguments and keyword arguments are passed to the function.
        """
        while True:
            try:
                output = f(*args, **kwargs)
            except tf.errors.ResourceExhaustedError:
                print("\nOut of memory, retrying...")
                if f == self.model.fit:
                    # Otherwise it will see some data more than once
                    print("Resetting to weights at epoch start")
                    self.reset_to_epoch_start()
                continue
            break
        return output

    def show_time(self, epoch):
        """Display time elapsed and remaining."""
        elapsed = time.time() - self.start_time # Start time is set at the start of epoch 0
        print("{:<25}{:<25}".format("Elapsed:", str(timedelta(seconds=np.round(elapsed)))))
        remaining = elapsed / (epoch + 1 - self.history.start_epoch) * (self.epochs + self.history.start_epoch - (epoch + 1))
        print("{:<25}{:<25}".format("Estimated remaining:", str(timedelta(seconds=np.round(remaining)))))
        return elapsed, remaining

    def show_progress_bar(self, epoch):
        """Display a progress bar for the training process."""
        # Bar has to be created every time because it gets cleared every epoch.
        bar = tqdm(range(self.history.start_epoch, self.history.start_epoch + self.epochs),
                   ncols=750, bar_format="{l_bar}{bar}{n}/{total}")
        bar.update(epoch - self.history.start_epoch + 1)
        bar.refresh()

    def get_score_pos(self, preds, data, dataset_name):
        """Calculate accuracy for the given PoS predictions (at word level)."""
        filtered_preds = preds[0].argmax(axis=-1).flatten()[self.eval_info[dataset_name]["real_tokens"]].tolist()
        filtered_logits = preds[0].reshape(
            (preds[0].shape[0] * preds[0].shape[1], preds[0].shape[2])
        )[self.eval_info[dataset_name]["real_tokens"]]
        new_preds = pos_utils.reconstruct_subwords(
            self.eval_info[dataset_name]["subword_locs"], filtered_preds, filtered_logits
        )

        assert len(new_preds) == len(self.eval_info[dataset_name]["all_labels"]), "Prediction and truth lengths do not match"
        return (np.array(self.eval_info[dataset_name]["all_labels"]) == np.array(new_preds)).mean()

    def get_score_sentiment(self, preds, data, dataset_name=None):
        """Calculate Macro F1 score for the given sentiment predictions."""
        return f1_score(data["sentiment"].values, preds[0].argmax(axis=-1),
                        average="macro", zero_division=0)

    def save_checkpoint(self, dev_score):
        """Save current weights as best weights."""
        print("\nDev score improved from", self.history.best_dev_score, "to", dev_score,
              ",\nsaving to " + self.checkpoint_filepath)
        self.model.save_weights(self.checkpoint_filepath)

    def manual_predict(self, data, batch_size):
        """Alternative prediction method, use when model.predict runs out of memory due to data size."""
        convert_functions = {("pos", "mbert"): data_preparation_pos.bert_convert_examples_to_tf_dataset,
                             ("pos", "xlm-roberta"): data_preparation_pos.roberta_convert_examples_to_tf_dataset,
                             ("sentiment", "mbert"): data_preparation_sentiment.bert_convert_examples_to_tf_dataset,
                             ("sentiment", "xlm-roberta"): data_preparation_sentiment.roberta_convert_examples_to_tf_dataset}
        preds = []
        if self.task == "pos":
            for i in tqdm(range(0, len(data), batch_size)):
                dataset = convert_functions[("pos", self.short_model_name)](
                    data[i:i+batch_size], self.tokenizer, self.tagset, self.max_length
                )
                dataset, batches = model_utils.make_batches(dataset, batch_size, repetitions=1, shuffle=False)
                preds.extend(self.model.predict(dataset, steps=batches)[0])
        else:
            for i in tqdm(range(0, data.shape[0], batch_size)):
                dataset = convert_functions[("sentiment", self.short_model_name)](
                    [(data_preparation_sentiment.Example(
                        text=text, category_index=label)
                     ) for label, text in data.values[i:i+batch_size]],
                    self.tokenizer, max_length=self.max_length
                )
                dataset, batches = model_utils.make_batches(dataset, batch_size, repetitions=1, shuffle=False)
                preds.extend(self.model.predict(dataset, steps=batches)[0])
        return (np.array(preds),) # For consistency

    def predict_all(self):
        """Return train and dev predictions."""
        if len(self.train_eval_data) < 1e5:
            train_preds = self.handle_oom(self.model.predict, self.train_eval_dataset,
                                          steps=self.train_eval_batches, verbose=1)
        else:
            # If train data is large, TF predict will always run out of memory
            train_preds = self.handle_oom(self.manual_predict, self.train_eval_data,
                                          batch_size=self.eval_batch_size)
        dev_preds = self.handle_oom(self.model.predict, self.dev_dataset,
                                    steps=self.dev_batches, verbose=1)
        return train_preds, dev_preds

    def train(self):
        """Train the model."""
        # Make sure class weights are calculated if they are needed
        if self.use_class_weights and not self.class_weights:
            print("Calculating class weights")
            self.calc_class_weights()
            print(self.class_weights)

        self.start_time = time.time()

        for epoch in range(self.history.start_epoch, self.history.start_epoch + self.epochs):
            print("\nEpoch", epoch)
            epoch_start = time.time()

            # Fit and evaluate
            hist = self.handle_oom(self.model.fit, self.train_dataset, epochs=1,
                                   steps_per_epoch=self.train_batches,
                                   class_weight=self.class_weights, verbose=1)
            loss = hist.history["loss"][0]
            print("Saving temp weights...")
            self.model.save_weights(self.temp_weights_filepath)
            train_preds, dev_preds = self.predict_all()

            # Show progress
            clear_output()
            elapsed, remaining = self.show_time(epoch)
            self.show_progress_bar(epoch)
            epoch_duration = time.time() - epoch_start

            # Calculate scores
            train_score = self.metric(train_preds, self.train_eval_data, "train_eval")
            dev_score = self.metric(dev_preds, self.dev_data, "dev")
            if dev_score > self.history.best_dev_score:
                self.save_checkpoint(dev_score)
                self.history.update_best_dev_score(train_score, dev_score,
                                                   epoch, epoch_duration, dev_preds)

            # Update and show history
            self.history.update_hist(epoch, loss, train_score, dev_score, epoch_duration)
            self.history.show_hist()
            self.history.plot()

    def make_definitive(self, delete_temp=False):
        """Rename all files linked to this checkpoint into their definitive names. Set
        'delete_temp' to True to delete the temporal weights file."""
        rename_files = [self.checkpoint_filepath, self.history.log_filepath,
                        self.history.checkpoint_params_filepath]
        if self.task == "sentiment":
            rename_files.append(self.history.checkpoint_report_filepath)
        for file in rename_files:
            os.replace(file, file.replace("_checkpoint", "").replace(self.suffix, ""))
        if delete_temp:
            os.remove(self.temp_weights_filepath)

    def compare_checkpoint(self):
        """Compare this Trainer's best dev score with other weight files from the same model,
        task and language. Print those that surpass it."""
        possible_weights = glob.glob(self.checkpoint_dir + "{}_{}*.hdf5".format(self.model_name,
                                                                                self.task))
        print("Current dev score: {:.4f}\n".format(self.history.best_dev_score))
        print("Weight files found:\n", *possible_weights, sep="\n")
        output = ""

        for weight_file in possible_weights:
            if weight_file != self.checkpoint_filepath:
                weight_file = re.split(r"\\|/", weight_file)[-1] # Get file name only
                unpacked = weight_file.split("_")

                # Get params file
                if len(unpacked) <= 2: # Definitive file, checkpoint is longer
                    param_file = re.findall(r"(.*)\.", weight_file)[0] + "_params.xlsx"
                else:
                    unpacked = unpacked[:3] + ["params"] + unpacked[3:]
                    param_file = re.findall(r"(.*)\.", "_".join(unpacked))[0] + ".xlsx"

                if os.path.isfile(self.history.logs_dir + param_file): # Check if the file exists first
                    df = pd.read_excel(self.history.logs_dir + param_file)
                    dev_score = df.set_index("Variable").loc["dev_score", "Value"]
                    if dev_score > self.history.best_dev_score:
                        output += "\n{} has a higher score of \033[1m{:.4f}\033[0m\n(taken from {})".format(
                            weight_file, dev_score, param_file
                        )
                else:
                    print("No 'params' file found for", weight_file)
        if output:
            print(output)
        else:
            print("\nNo better weights found.")

    def delete_checkpoint(self):
        """Deletes checkpoint's weights and files, along with temporary weights."""
        delete = [self.checkpoint_filepath,
                  self.temp_weights_filepath,
                  self.history.log_filepath,
                  self.history.checkpoint_params_filepath,
                  self.history.checkpoint_report_filepath]

        for file in delete:
            if os.path.isfile(file):
                print("Removing {}...".format(file))
                os.remove(file)

    def get_main_params(self):
        """Return dictionary with the Trainer's main parameters."""
        include = ["training_lang", "data_path", "task", "use_class_weights",
                   "model_name", "max_length", "train_batch_size", "eval_batch_size",
                   "learning_rate", "epochs", "num_labels", "checkpoint_filepath"]
        return {k: v for k, v in self.__dict__.items() if k in include}

class History:
    """Stores training history and handles log files."""
    def __init__(self, trainer):
        # Dirs and files
        self.logs_dir = trainer.checkpoint_dir + "logs/"
        self.log_filepath = self.logs_dir + "{}_{}_checkpoint_log{}.xlsx".format(trainer.model_name,
                                                                                 trainer.task,
                                                                                 trainer.suffix)
        self.checkpoint_params_filepath = self.logs_dir + "{}_{}_checkpoint_params{}.xlsx".format(trainer.model_name,
                                                                                                  trainer.task,
                                                                                                  trainer.suffix)
        self.checkpoint_report_filepath = self.logs_dir + "{}_{}_checkpoint_report{}.xlsx".format(trainer.model_name,
                                                                                                  trainer.task,
                                                                                                  trainer.suffix)
        if not os.path.isdir(self.logs_dir):
            os.makedirs(self.logs_dir)

        # History attributes
        self.epoch_list = []
        self.loss_list = []
        self.train_score_list = []
        self.dev_score_list = []
        self.total_time_list = []
        self.start_epoch = 0
        self.best_dev_score = 0
        self.best_dev_epoch = None
        self.best_dev_total_time = None

        # Other
        self.task = trainer.task
        self.dev_data = trainer.dev_data
        self.metric_name = metric_names[self.task]
        self.trainer_params = trainer.get_main_params()

    def load_from_checkpoint(self):
        """Load history from last checkpoint."""
        log = pd.read_excel(self.log_filepath)
        end_index = log["dev_score"].argmax() + 1 # Get history until best dev
        index_best_dev_score = end_index - 1
        self.epoch_list = log["epoch"].values[:end_index].tolist()
        self.loss_list = log["loss"].values[:end_index].tolist()
        self.train_score_list = log["train_score"].values[:end_index].tolist()
        self.dev_score_list = log["dev_score"][:end_index].values.tolist()
        self.total_time_list = log["total_time"][:end_index].values.tolist()
        self.best_dev_score = self.dev_score_list[-1]
        self.best_dev_epoch = self.epoch_list[-1]
        self.start_epoch = self.epoch_list[-1] + 1
        print("Checkpoint dev score:", self.best_dev_score)

    def load_from_last(self):
        """Load history from last epoch."""
        log = pd.read_excel(self.log_filepath)
        index_best_dev_score = log["dev_score"].argmax()
        self.epoch_list = log["epoch"].values.tolist()
        self.loss_list = log["loss"].values.tolist()
        self.train_score_list = log["train_score"].values.tolist()
        self.dev_score_list = log["dev_score"].values.tolist()
        self.total_time_list = log["total_time"].values.tolist()
        self.best_dev_score = self.dev_score_list[index_best_dev_score]
        self.best_dev_epoch = self.epoch_list[index_best_dev_score]
        self.start_epoch = self.epoch_list[-1] + 1
        print("Best dev score:", self.best_dev_score)

    def convert_time(self, t):
        """Convert seconds to standard time format."""
        return str(timedelta(seconds=np.round(float(t))))

    def update_best_dev_score(self, train_score, dev_score, epoch, epoch_duration, dev_preds):
        """Update attributes and files with a new best validation score."""
        self.best_dev_score = dev_score
        self.best_dev_epoch = epoch
        if self.total_time_list:
            new_time = self.total_time_list[-1] + epoch_duration
        else:
            new_time = epoch_duration
        self.best_dev_total_time = self.convert_time(new_time)
        # Parameters with which the score was obtained
        params = {**self.trainer_params,
                  **{"epoch": epoch, "train_score": train_score, "dev_score": dev_score,
                     "total_training_time": self.best_dev_total_time}}
        pd.DataFrame(params.items(), columns=["Variable", "Value"]).to_excel(
            self.checkpoint_params_filepath, index=False
        )
        # Classification report for sentiment
        if self.task == "sentiment":
            report = classification_report(self.dev_data["sentiment"].values,
                                           dev_preds[0].argmax(axis=-1), output_dict=True)
            pd.DataFrame(report).transpose().to_excel(
                self.checkpoint_report_filepath
            )

    def update_hist(self, epoch, loss, train_score, dev_score, epoch_duration):
        """Update history with data from a new epoch."""
        self.epoch_list.append(epoch)
        self.loss_list.append(loss)
        self.train_score_list.append(train_score)
        self.dev_score_list.append(dev_score)
        if self.total_time_list:
            new_time = self.total_time_list[-1] + epoch_duration
        else:
            new_time = epoch_duration
        self.total_time_list.append(new_time)

        pd.DataFrame({"epoch": self.epoch_list,
                      "loss": self.loss_list,
                      "train_score": self.train_score_list,
                      "dev_score": self.dev_score_list,
                      "total_time": self.total_time_list,
                      "total_time_h:m:s": [self.convert_time(t) for t in self.total_time_list]}
                      ).to_excel(self.log_filepath, index=False)

    def show_hist(self):
        """Print history in table format."""
        print("\nHistory:\n")
        print("Best dev score so far: \033[1m{:.4f}\033[0m\n".format(self.best_dev_score))
        print("{:<20}{:<20}{:<20}{:<20}".format("Epoch", "Loss", "Train score", "Dev score"))
        for epoch in self.epoch_list:
            if epoch == self.best_dev_epoch:
                bold_code = ("\033[1m", "\033[0m") # Add bold to row where best dev score was found
            else:
                bold_code = ("", "")
            print(bold_code[0] + "{:<20}{:<20.4f}{:<20.4f}{:<20.4f}".format(
                  self.epoch_list[epoch], self.loss_list[epoch], self.train_score_list[epoch],
                  self.dev_score_list[epoch]) + bold_code[1]
            )

    def plot(self):
        """Show loss and score history plots."""
        sns.set()
        sns.set_style({"axes.linewidth": 1, "axes.edgecolor": "black",
                       "xtick.bottom": True, "ytick.left": True})
        fig, ax = plt.subplots(1, 2, figsize=(12,4))
        plt.subplots_adjust(wspace=0.15)
        ax[0].plot(self.epoch_list, self.loss_list)
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[1].plot(self.epoch_list, self.train_score_list)
        ax[1].plot(self.epoch_list, self.dev_score_list)
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel(self.metric_name)
        ax[1].legend(["Train", "Dev"])
        sns.despine()
        plt.show()
        plt.close()

    def get_best_dev(self):
        """Return best validation score, and the epoch and training time at which it was obtained."""
        return self.best_dev_score, self.best_dev_epoch, self.best_dev_total_time
