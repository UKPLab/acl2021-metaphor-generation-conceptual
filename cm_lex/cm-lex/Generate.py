import sys, getopt

from LexicalReplacement import LexicalReplacement
from ModelTesting import Test


config = {"mappings":"../mappings/w2v_50.json",
          "embeddings":"../models/w2v_50"}


def text_to_tests(texts):
    for line in texts:
        yield Test.from_tsv(line)


def generate(model, texts):
    tests = text_to_tests(texts)

    for t in tests:
        print(model.generate_test(t))
    
    
def main(argv=None):
    if argv is None:
        argv = sys.argv

    try:
        opts, args = getopt.getopt(argv[1:], "hm:e:", ["help", "mappings", "embeddings"])
    except Exception as e:
        print ("Error in args : " + str(argv[1:]))
        print (e)
        return 2


    for o in opts:
        if o[0] == "-m":
            config["mappings"] = o[1]
        if o[0] == "-e":
            config["embeddings"] = o[1]

    model = LexicalReplacement(config["mappings"], config["embeddings"])

    texts = [t.split("\t") for t in open(args[0]).readlines()]
    generate(model, texts)


if __name__ == "__main__":
    main()
