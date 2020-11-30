Typology of cross-lingual approaches
==============

The idea is to explore the effect of morphological typology (fusional, isolating, agglutinative or introflexive) on cross-lingual approaches on a more syntactic task (POS Tagging or Dependency Parsing) and a more semantic task (sentiment analysis). The broad question is: what effect does morphological typology have on cross-lingual approaches?

Languages
----
- Fusional: Bulgarian (bl), Croatian (hr), English (en), Russian (ru), Slovak (sl), German (de), Spanish (es), Norwegian (no), Greek (el)
- Isolating: Chinese (zh), Vietnamese (vi), Thai (th), Cantonese (zh-yue), Bahasa Indonesia (id)
- Agglutinative: Basque (eu), Finnish (fi), Japanese (ja), Korean (ko), Turkish (tr)
- Introflexive: Modern Standard Arabic (ar), Hebrew (he), Algerian (ar-dz), Maltese (mt)


Data
----

We have POS and dependency annotated data for all langauges available from the [Universal Dependencies project](https://universaldependencies.org/). These datasets are in the standard conllu format, which includes one token per line, with tab separated fields, and sentences separated by a new line.

The fields are:
1. word id
2. form
3. lemma
4. universal part-of-speech tag
5. language specific part-of-speech-tag
6. morphological features
7. dependency head
8. dependency relation
9. enhanced dependency graph
10. miscellaneous

Example:
```
# newdoc id = n01018
# sent_id = n01018024
# text = It's like a super power sometimes.
1	It	it	PRON	PRP	Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs	6	nsubj	6:nsubj	SpaceAfter=No
2	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	6	cop	6:cop	_
3	like	like	ADP	IN	_	6	case	6:case	_
4	a	a	DET	DT	Definite=Ind|PronType=Art	6	det	6:det	_
5	super	super	ADJ	JJ	Degree=Pos	6	amod	6:amod	_
6	power	power	NOUN	NN	Number=Sing	0	root	0:root	_
7	sometimes	sometimes	ADV	RB	_	6	advmod	6:advmod	SpaceAfter=No
8	.	.	PUNCT	.	_	6	punct	6:punct	_
```



We also have similar sentiment datasets for these languages in csv format, where the first element is the label (0=negative, 1=positive) and the second element is the text.


Example:
```
1,It was better than expected.
0,The service was terrible.
```



Each dataset is divided into train.csv, dev.csv, test.csv. The data can be extracted into python using the csv class.

```
import csv

labels, texts = [], []

with open("data/sentiment/en/train.csv") as infile:
	for label, text in csv.reader(infile):
		labels.append(label)
		texts.append(text)
```


Current Problems
----
1. Need to deal with longer sentiment examples by breaking into chunks that fit into transformer model and then aggregating the predictions
