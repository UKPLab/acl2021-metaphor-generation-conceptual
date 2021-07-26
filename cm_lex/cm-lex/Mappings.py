import sys, getopt
import json

import Embedding
import Frames

# Default parameters
config = {"model_type":"w2v",
          "model_file":"../models/w2v_50",
          "output_file":"../mappings/w2v_50.json"}


def known_mappings():
    """
    Dictionary of all frame->frame mappings based on the available frames
    :return:
    """
    mappings = {}
    for frame in Frames.frames:
        for frame2 in Frames.frames:
            mappings[frame.upper() + " " + frame2.upper()] = None 
    return mappings


def learn_frame_mappings(input_embeddings, output_file):
    """
    Using an embedding model, generate the vectors corresponding to the mappings between all known frames.

    :param input_embeddings:
    :param output_file:
    :return:
    """
    km = known_mappings()
    final_mappings = {k:None for k in km}

    for mapping in [k.split() for k in km.keys()]:
        if mapping[0] in input_embeddings.wv and mapping[1] in input_embeddings.wv:
            final_mappings[mapping[0] + " " + mapping[1]] = [float(v) for v in (input_embeddings.wv[mapping[1]] - input_embeddings.wv[mapping[0]])]
    
    json.dump(final_mappings, open(output_file, "w"))


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        opts, args = getopt.getopt(argv[1:], "hm:t:", ["help"])
    except Exception as e:
        print ("Error in args : " + str(argv[1:]))
        print (e)
        return 2

    model = None

    for o in opts:
        if o[0] == "-m":
            config["model_type"] = o[1]
        if o[0] == "-i":
            config["model_file"] = o[1]
        if o[0] == "-o":
            config["output_file"] = o[1]

    if config["model_type"] == "w2v":
        model = Embedding.load_w2v(config["model_file"])
    elif config["model_type"] == "ft":
        model = Embedding.load_fasttext(config["model_file"])
    #    elif config["model_type"] == "glove":
    #        model = Embedding.load_glove(model_file)

    learn_frame_mappings(model, config["output_file"])


if __name__ == "__main__":
    main()

