from nltk.corpus import framenet as fn
from nltk.stem import WordNetLemmatizer
import json

'''
Object for identifying potential source and target frames
'''
class FnFrames(object):
    def __init__(self):
        self.load_verb_frame_dict()
        self.lemmatizer = WordNetLemmatizer()
        self.mapping_frequencies = json.load(open("../data/mapping_training_frequencies.json"))     # built from training data

    # Dictionary of verb -> potential frames in FN, using NLTK API
    def load_verb_frame_dict(self):
        self.verb_frame_dict = {}
        for frame in fn.frames():
            for lu in frame.lexUnit:
                if lu.endswith(".v"):
                    verb = lu[:-2]
                    if verb not in self.verb_frame_dict:
                        self.verb_frame_dict[verb] = []
                    self.verb_frame_dict[verb].append(frame.name)

    # Given a sentence, return all possible potential target domains
    # Returns a dict of word index:list of possible frames
    # You can use your own tokenization, or it will just split - more advanced tokenization is always possible
    # Currently finds domains for all words, could also add a parser to only look at a verbs
    def get_possible_target_frames(self, sentence, focus_word_index=None):
        res = {}
        if type(sentence) != list:
            sentence = [self.lemmatizer.lemmatize(word, "v") for word in sentence.split()]
        if focus_word_index:
            focus_word = sentence[focus_word_index]
            res = {focus_word:self.verb_frame_dict[focus_word]}
        else:
            for c, word in enumerate(sentence):
                if word in self.verb_frame_dict:
                    res[c] = self.verb_frame_dict[word]
        return res

    # Given a target domain, find all the source domains that align
    # By default returns top 3, but this can be changed: many may have only 3 or 4, making this not the best
    # Use reverse to get rare frames instead
    def get_possible_source_frames(self, target, top_n=3, reverse=True):
        target = target.upper()

        if target not in self.mapping_frequencies:
            return None
        else:
            return sorted(self.mapping_frequencies[target].items(), key=lambda item: item[1], reverse=reverse)[:top_n]

    # helper to get all possible frames that occur in the target-domain sentences
    def get_all_target_frames(self):
        return list(self.mapping_frequencies.keys())

    # helper to get all possible frames that occur in the source-domain sentences
    def get_all_source_frames(self):
        frames = set()
        for key in self.mapping_frequencies:
            frames |= self.mapping_frequencies[key].keys()
        return list(frames)
