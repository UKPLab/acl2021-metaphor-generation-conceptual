from LexicalReplacement import LexicalReplacement


config = {"mappings":"../mappings/w2v_50.json",
          "embeddings":"../models/w2v_50"}

class Test(object):
    """
    Test objects built for evaluating the model. You can format your data to match ../tests/final.tsv, and use
    this class to pass the tests to the generator.

    """
    def __init__(self, text, source=None, target=None, focus_id=None, gold=None):
        self.text = text
        self.source = source
        self.target = target
        self.focus_id = focus_id
        self.gold = gold

    @staticmethod
    def from_tsv(data, target_sent=True):
        """
        Helper to generate tests from a tab-separated line
        :param line:
        :return:
        """

        if target_sent:
            targ = data[3]
        return Test(data[2], data[-2], data[-1].strip(), int(data[4]), data[3])

    def __repr__(self):
        return self.text + " " + self.source + " " + self.target + " " + self.focus_id + " " + self.gold
    

def test_model(texts):
    """
    Run all the tests on the test data
    :param texts:
    :return:
    """
    model = LexicalReplacement(config["mappings"], config["embeddings"])

    for text in texts:
        print(model.generate_test(Test.from_tsv(text)))

    
def main(argv=None):
    texts = [t.split("\t") for t in open("tests/final.tsv").readlines()]
    test_model(texts)


if __name__ == "__main__":
    main()
