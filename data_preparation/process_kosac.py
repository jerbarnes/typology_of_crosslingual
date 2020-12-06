import csv
from collections import Counter
import numpy as np

def aggregate(polarities):
    c = Counter(polarities)
    if "NEG" not in c and "POS" not in c:
        return "NEU"
    if c["NEG"] > c["POS"]:
        return "NEG"
    if c["POS"] > c["NEG"]:
        return "POS"
    if c["POS"] == c["NEG"]:
        return "Conflict"

dataset = {}

with open("kosac-corpus-130808.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quotechar='"')
    for i, row in enumerate(reader):
        if i != 0:
            tag_id, sent_id, tag_type, morphemes, otype, subjectivity_type, subjectivity_polarity, polarity, intensity, nested_source, target, comment, confident, raw_sentence, sentence_morph = row
            sent_id = int(sent_id)
            if sent_id in dataset:
                dataset[sent_id]["polarities"].append(polarity)
            else:
                dataset[sent_id] = {}
                dataset[sent_id]["text"] = raw_sentence
                dataset[sent_id]["polarities"] = [polarity]

for sent_id in dataset.keys():
    agg = aggregate(dataset[sent_id]["polarities"])
    dataset[sent_id]["aggregate"] = agg


pos, neg = [], []
for sent_id in dataset.keys():
    text = dataset[sent_id]["text"]
    if dataset[sent_id]["aggregate"] == "POS":
        pos.append([1, text])
    if dataset[sent_id]["aggregate"] == "NEG":
        neg.append([0, text])

train_split, dev_split = 0.7, 0.8

pos_train_split = int(len(pos) * train_split)
pos_dev_split = int(len(pos) * dev_split)
train_pos = pos[:pos_train_split]
dev_pos = pos[pos_train_split:pos_dev_split]
test_pos = pos[pos_dev_split:]

neg_train_split = int(len(neg) * train_split)
neg_dev_split = int(len(neg) * dev_split)
train_neg = neg[:neg_train_split]
dev_neg = neg[neg_train_split:neg_dev_split]
test_neg = neg[neg_dev_split:]

train = train_pos + train_neg
dev = dev_pos + dev_neg
test = test_pos + test_neg

np.random.shuffle(train)
np.random.shuffle(dev)
np.random.shuffle(test)

with open("train.csv", "w") as outfile:
    writer = csv.writer(outfile)
    for row in train:
        writer.writerow(row)

with open("dev.csv", "w") as outfile:
    writer = csv.writer(outfile)
    for row in dev:
        writer.writerow(row)

with open("test.csv", "w") as outfile:
    writer = csv.writer(outfile)
    for row in test:
        writer.writerow(row)
