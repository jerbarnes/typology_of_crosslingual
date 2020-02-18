Typology of cross-lingual approaches
==============

The idea is to explore the effect of morphological typology (fusional, isolating, agglutinative or introflexive) on cross-lingual approaches on a more syntactic task (POS Tagging or Dependency Parsing) and a more semantic task (sentiment analysis). The broad question is: what effect does morphological typology have on cross-lingual approaches?

Languages
----
Fusional: Bulgarian, Croatian, English, Russian, Slovak
Isolating: Chinese, Vietnamese, Thai
Agglutinative: Basque, Finnish, Japanese, Korean, Turkish
Introflexive: Arabic, Hebrew


Data
----

We have POS and dependency annotated data for all langauges available from the [Universal Dependencies project](https://universaldependencies.org/)

We also have similar sentiment datasets for these languages.


Current Problems
----
1. We still need to properly binarize some of the data (ko,ru,vi).
2. We need to make the training and test datasets comparable.
