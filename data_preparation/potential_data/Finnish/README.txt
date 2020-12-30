NAME: FinnSentiment, source
LICENSE: This corpus is licensed with CC BY 4.0

For more information see http://urn.fi/urn:nbn:fi:lb-2020111001

FinnSentiment is a Finnish social media corpus for sentiment polarity annotation. 27,000 sentence data set annotated independently with sentiment polarity by three native annotators.

The creation of the corpus is documented in K. Lindén, T. Jauhiainen, S. Hardwick (2020): FinnSentiment - A Finnish Social Media Corpus for Sentiment Polarity Annotation. arXiv 2020.

The corpus is available in a utf-8 encoded TSV (tab-separated values) file with columns as indicated in the following list. In the list, "split" refers to the cross-validation split to which a sentence belongs, and "batch" to the work package the sentence belongs to. Indexes to the original corpus are strings consisting of a filename, like comments2008c.vrt, a space character, and a sentence id number in the file.

Column	Column name				Range / data type
1 	A sentiment				[-1, 1]
2 	B sentiment				[-1, 1]
3 	C sentiment				[-1, 1]
4 	majority value				[-1, 1]
5 	derived value				[1, 5]
6 	pre-annotated sentiment smiley		[-1, 1]
7 	pre-annotated sentiment product review	[-1, 1]
8 	split # 				[1, 20]
9 	batch # 				[1,9]
10 	index in original corpus		Filename & sentence id
11 	sentence text				Raw string