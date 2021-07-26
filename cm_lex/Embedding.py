import numpy as np
import heapq
import torch
import pickle

from sklearn import preprocessing
from transformers import BertModel
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords

from FNTokenizer import FNTokenizer
import Frames
#from glove import Corpus, Glove

config = {"model":BertModel,
          "tokenizer":FNTokenizer}

LIMIT = 1000


def load_fasttext(input_file):
    return FastText.load(input_file)


def load_w2v(input_file):
    return Word2Vec.load(input_file)


def load_bbc(in_file):
    vecs = pickle.load(open(in_file + ".m", "rb"))
    vocab = pickle.load(open(in_file + ".iw", "rb"))

    return Embedding(vecs, vocab)


# Some issues with glove python package, working on...
'''
def load_glove(input_file):
    model = Glove.load(input_file)
    vocab = list(model.dictionary.keys())
    vecs = np.array([model.word_vectors[model.dictionary[v]] for v in vocab])
    return Embedding(vecs, vocab)
'''


class Embedding(object):
    """
    Base class for all embeddings.
    Based heavily on the Embedding class from https://github.com/williamleif/histwords
    """

    @staticmethod
    def from_bert_model_and_text(vec_model, texts, output_file=None, agg=np.mean):
        """
        Given a (transformer) embedding model and a text, return an embedding space at the word level.
        Takes the embedding for each word found in the text, and combines them using the agg function.
        """
        vocab, vecs = None, None
        model = BertModel.from_pretrained(vec_model)

        # Need special tokenizer to keep frames as their own tokens
        tokenizer = FNTokenizer.from_pretrained("bert-base-cased")
        tokenizer.add_tokens([f.upper() for f in Frames.frames])

        embeds = {}

        for c, line in enumerate(texts):
            input_ids = torch.tensor(tokenizer.encode(line)).unsqueeze(0)
            outputs = model(input_ids)
            last_hidden_states = outputs[0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            for i, token in enumerate(tokens):
                if token not in stopwords.words('english'):
                    embed = last_hidden_states[0][i].detach().numpy()
                    if token not in embeds:
                        embeds[token] = []
                    if len(embeds[token]) < LIMIT:
                        embeds[token].append(embed)

            if (c+1) % 1000 == 0 or c == len(texts)-1:
                print (c, len(texts), len(embeds), np.mean([len(embeds[v]) for v in embeds]))
                types = list(embeds.keys())
                res = {}
                for key in embeds:
                    res[key] = agg(embeds[key], axis=0)

                vocab = sorted(list(res.keys()))
                vecs = np.array([res[v] for v in vocab])
                if (c+1) % 10000 == 0 and output_file:
                    e = Embedding(vecs, vocab)
                    e.save("embeddings/" + output_file + str((c+1)/10000))
        return Embedding(vecs, vocab)

    @staticmethod
    def new_init(input_file, normalize=True, **kwargs):
        """
        Alternate method for initialization which uses gensim formating.

        :param input_file:
        :param normalize:
        :param kwargs:
        :return:
        """
        vecs, vocab = [], []
        for line in open(input_file):
            vocab.append(line.split()[0])
            vecs.append(line.split()[1:])

        vecs = np.array(vecs, dtype="float")
        vocab = set(vocab)
        return Embedding(vecs, vocab, normalize)

    def __init__(self, vecs, vocab, normalize=True, **kwargs):
        """
        Normal initialization, which requires all the parameters

        :param vecs: matrix of vectors
        :param vocab: vocabulary
        :param normalize:
        :param kwargs:
        """
        self.m = vecs
        self.dim = self.m.shape[1]
        self.iw = vocab
        self.wi = {w:i for i,w in enumerate(self.iw)}
        if normalize:
            self.normalize()

    def __getitem__(self, key):
        if self.oov(key):
            raise KeyError
        else:
            return self.represent(key)

    def __iter__(self):
        return self.iw.__iter__()

    def __contains__(self, key):
        return not self.oov(key)

    def normalize(self):
        preprocessing.normalize(self.m, copy=False)

    def oov(self, w):
        return not (w in self.wi)

    def represent(self, w):
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            print ("OOV: ", w)
            return np.zeros(self.dim)

    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        sim = self.represent(w1).dot(self.represent(w2))
        return sim

    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w))
        return heapq.nlargest(n, zip(scores, self.iw))

    def save(self, out_file):
        pickle.dump(self.m, open(out_file + ".m", "wb"))
        pickle.dump(self.iw, open(out_file + ".iw", "wb"))
