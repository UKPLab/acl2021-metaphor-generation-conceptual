from nltk.corpus import framenet as fn
import Embedding
import random
from fitbert import FitBert
import json


class Lexical(object):
    """
    Root class for lexical generation. We experimented with some other models, which this can be used as the base for.
    """
    def generate_for_sample(self, sample):
        return self.generate(sample.text, sample.source, sample.target, sample.focus_id)


class LexicalReplacement(Lexical):
    """
    Main Lexical Replacement model from the paper.

    It initializes with frame-based embeddings and mappings, and given a sample it generates a metaphor that evokes
    the source/target mapping. Requires text, source/target domains, and the id of the focus word.

    If the word isn't in the embeddings, or can't be mapped, returns None
    """

    def __init__(self, mappings, embeddings):
        self.mapping = json.load(open(mappings))
        self.embeddings = Embedding.load_w2v(embeddings)
        self.fb = FitBert()

    def generate_test(self, test):
        """
        Generates based on Test instances from the ModelTesting class
        :param test:
        :return:
        """
        return self.generate(test.text, test.source, test.target, test.focus_id)

    def generate(self, input_sentence, source, target, focus_id):
        """
        Generates using full inputs
        :param input_sentence:
        :param source:
        :param target:
        :param focus_id:
        :return:
        """
        source_frame = source.upper()
        target_frame = target.upper()

        input_words = input_sentence.split()
        focus_word = input_words[int(focus_id)]
        input_words[int(focus_id)] = "***mask***"

        mapping = source_frame + " " + target_frame
        if mapping not in self.mapping or not self.mapping[mapping]:
            filled_string = "None"
        else:
            mapping = self.mapping[mapping]    
            if focus_word not in self.embeddings:
                filled_string = None
            else:
                mapped = [v[0] for v in self.embeddings.most_similar(positive=[self.embeddings.wv[focus_word] + mapping], topn=100)]
                mapped = [w for w in mapped if not w.isupper()][:5]
                text = " ".join(input_words)

                if not len(mapped):
                    filled_string = None
                else:
                    filled_string = self.fb.fitb(text, mapped, delemmatize=True)
        if filled_string == "None":
            return None
        else:
            return filled_string

