# -*- coding: utf-8 -*-
import sys, os, getopt
import random
import logging

from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
#from glove import Corpus, Glove

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.INFO)

config = {"input_file":"../data/frame_corpus.txt",
          "output_file":"../models/w2v_50",
          "model_type":"w2v",
          "dimensions":50}

'''
def load_glove(input_lines, size=100, iter=20):
    corpus = Corpus()
    corpus.fit(input_lines)

    glove = Glove(no_components=size)
    glove.fit(corpus.matrix, epochs=iter)
    glove.add_dictionary(corpus.dictionary)
    return glove
'''


def iterative_train_model(input_file, output_dir, embedding_class, d=100):
    """
    Training a given vector model

    :param input_file:
    :param output_dir:
    :param embedding_class:
    :param d:
    :return:
    """
    data = [line.split() for line in open(input_file, encoding="utf-8")]
    random.shuffle(data)
    
    model = embedding_class(data, size=d, iter=2)
    model.save(output_dir)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:m:d:", ["help", "input", "output", "model", "dimensions"])
    except Exception as e:
        print ("Error in args : " + str(argv[1:]))
        print (e)
        return 2

    model = None

    for o in opts:
        if o[0] == "-i":
            config["input_file"] = o[1]
        if o[0] == "-o":
            config["output_file"] = o[1]
        if o[0] == "-m":
            config["model_type"] = o[1]
        if o[0] == "-d":
            config["dimensions"] = int(o[1])

    if model_type == "w2v":
        model = Word2Vec
    elif model_type == "ft":
        model = FastText
#    elif model_type == "glove":
#        model = load_glove

    print ("Training embeddings on input {input_file} using model {model_type} of size {dimensions}".format(input_file=input_file, model_type=model_type, dimensions=dimensions))
    iterative_train_model(config["input_file"], config["output_file"], model, d=config["dimensions"])


if __name__ == "__main__":
    sys.exit(main())
