#!/usr/bin/env python3
"""
Creates and trains a gensim word2vec model
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """Creates and trains a gensim wrod2vec model
    sentences: a list of sentences to be trained on
    size: the dimensionality of the embedding layer
    for use in training
    window: the max distance between the current and
    predicted word within a sentence
    negative: the size of negative sampling
    cbow: boolean to determine the training type
    min_count: the min number of ocurrences of a word;
    True is for cbow and false is for skip-gram
    iterations: the size of negative sampling
    seed: the seed for the random number generator
    workes: the number of worker threads to train the
    model"""
    if cbow is True:
        skip = 0
    else:
        skip = 1
    model = Word2Vec(size=size, window=window,
                     min_count=min_count, seed=seed)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=iterations)
    return model
