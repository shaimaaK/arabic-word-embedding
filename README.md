<div align="center">
  <img  src="https://github.com/shaimaaK/arabic-word-embedding/assets/54285485/8b52c843-12e2-4de3-ab13-e9abb3d56d24">
 </div>

# Arabic Word Embedding
THis project is implemented as part of the "Natural Language Processing and Deep learning " course during my master's degree. In this project I have created two word embedding models: Word2Vec-SkipGram and GLoVE using [ArWiki Dump 2018 Dataset](https://www.kaggle.com/datasets/abedkhooli/arabic-wiki-data-dump-2018) where the Skipgram model is imporved by tuning the values for vector size and window.

## Table of contents
- [Workflow](#workflow)
  * [Step 1 preprocessing](#step-1-preprocessing)
  * [Step 2 define parameters for tuning](#step-2-define-parameters-for-tuning)
  * [Step 3 build the word embedding models](#step-3-build-the-word-embedding-models)
  * [Step 4 evaluate the performance](#step-4-evaluate-the-performance)
- [Requirements](#requirements)
- [References and Resources](#references-and-resources)

## Workflow
### Step 1 preprocessing 
**Step 1.1: reading the corpus** <br>
Parse the compressed arabic wiki articles of the format `.bz2` using the Gensim utility `WikiCorpus` then make sure the encoding is set to **utf-8** as arabic language is encoded as the latin based encoding : utf-8 <br>
**Step 1.2: remove unwanted characters from the scanned articles** <br>
- Non-arabic character (mainly english in upper and lower case ) 
- Digits [0-9]
- Extra spaces
- Tashkeel and tatweel (arabic diacritics) </ul>
**Step 1.3: save the output to corpus_cleaned.txt** <br>
The output of the preprocessing is stored hence is this step to generate the cleaned data is only executed once. Note the corpus_cleaned.txt is omitted from the repository as it exceeds the allowed size for github repository

### Step 2 define parameters for tuning
*gensim.models.Word2Vec(sentences,vector_size,window,sg,workers)*<hr>
- List of used Training parameters:
    - **sentences** : corpus of the dataset
    - **vector_size** (int, optional) – Dimensionality of the word vectors.
    - **window** :(int, optional) – Maximum distance between the current and predicted word within a sentence
    - **sg** ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW
    - **workers** (int, optional) – Use these many worker threads to train the model (=faster training with multicore machines). 
- Summary of list of parameters needs tuning : 1) vector_size, 2) window-size 3) learning rate
- (SkipGram) List of parameters tuning:
  - vector_size_list=[500,1000]
  - window_list=[10,15,20]
- (GLoVE) List of parameters tuning
  - learning rate=[0.01,0.05]
  - window_list=[10,15,20] </ul>
As common knowledge in the NLP research community the window size starts from 5, therefore we have tried 10,15,20 on SkipGram and on GLoVE we tired 10,15. Another parameter dependent on the training corpus, is the embedding matrix size which is tested as 500, and 1000. Unfortunately, (as expected) the value 1000 generated a memory error as the environment memory is unable to allocate the enough space to run either of the algorithms therefore is fixed to 500. Lastly some parameters are dedicated to GLoVE are also
experimented with such as the learning rate 0.01 and 0.05 while epochs parameter is fixed to 50 to avoid extensive runtime. It worth mentioning, the experiments are tested on 3 variations of SkipGram model and 4 varations of GLoVE model and the results discussed are chosen from the full results which are available and can be tested using the provided code.

### Step 3 build the word embedding models
Using GenSim and GLoVE libararies on python the arabic-word-embedding models are trained and saved. It is woth noting that the GLoVE library only worked on Colab with older versions of python (3.7 and lower) as the library implementation is developed for those version of python

### Step 4 evaluate the performance
**Test 1 : Most Similar Words** <br>
Find the top-N most similar words. Positive words contribute positively towards the similarity, negative words negatively. [link](https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.most_similar.html#gensim.models.Word2Vec.most_similar)<br>
- Pick 8 Arabic words and, for each one, ask each model about
the **most similar 10 words to it. Plot the results using t-SNE** (or scatterplot) and discuss
them

**Test 2: Odd-One-Out**<br>
- we ask our model to give us the word that does not belong to the list [doc](https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.doesnt_match.html#gensim.models.Word2Vec.doesnt_match)
- Pick **5-10 triplets** of Arabic words and, for each one, ask each model to
pick the word in the triplet that does not belong to it. Discuss the results.

**Test 3: Measuring Sentence Similarity**<br>
Find the Sentences similar to each other by computing the **cosine similarity** function of the two embedding vectors as in [Paul Minogue blog](https://paulminogue.com/index.php/2019/09/29/introduction-to-cosine-similarity/)<br>
write **5 sentences in Arabic**. For each sentence, **pick 2-3
words and replace them with their synonyms or antonyms**. Use your embeddings to
compute the similarity between each sentence and its modified version. Discuss the
results

**Test 4: Analogy**<br>
- Syntax in [link](https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.doesnt_match.html#gensim.models.Word2Vec.doesnt_match)
- pick 5-10 cases of analogies in Arabic, like the one we used in class:


## Requirements
- glove-python-binary
- arabic_reshaper
- python-bidi
- pyarabic.araby
- gensim
- matplotlib.pyplot
- seaborn
- sklearn.manifold.TSNE

## References and Resources
All references and resources are documented used in each step are documented in the .ipynb file in markdown
