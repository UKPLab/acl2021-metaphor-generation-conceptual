import sys, getopt
import torch
import os
import time
import numpy as np

from fairseq.models.bart import BARTModel

os.environ['CUDA_VISIBLE_DEVICES']="1"
config = {"input":"data/test/knownmapping.input",
          "checkpoint_dir":"checkpoints/",
          "checkpoint_file":"checkpoint_best.pt"}


np.random.seed(4)
torch.manual_seed(4)

def generate_from_file(input_path):
    bsz = 1
    temp = 0.7
    top_k = 5

    try:
        bart = BARTModel.from_pretrained(config["checkpoint_dir"], checkpoint_file=config["checkpoint_file"], bpe="gpt2")
    except OSError as e:
        print ("OSError, check that the trained checkpoint exists: " + config["checkpoint_dir"] + config["checkpoint_file"])
        return 1

    # Comment out to use CPU
    #bart.cuda()
    bart.eval()

    with open(input_path) as input_file, open(input_path + '.hypo', 'w') as fout:
        line = input_file.readline().strip()
        batch_lines = [sline]

        ## Batch generation
        for count, line in enumerate(input_file):
            batch_lines.append(line)
            if count + 1 % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(batch_lines,
                                                   sampling=True,
                                                   sampling_topk=top_k,
                                                   temperature=temp,
                                                   lenpen=2.0,
                                                   max_len_b=30,
                                                   min_len=7,
                                                   no_repeat_ngram_size=3)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis.replace('\n','') + '\n')
                    fout.flush()
                batch_lines = []

        ## Extras at the end get their own batch
        if batch_lines != []:
            hypotheses_batch = bart.sample(batch_lines,
                                           sampling=True,
                                           sampling_topk=top_k,
                                           temperature=temp,
                                           lenpen=2.0,
                                           max_len_b=30,
                                           min_len=7,
                                           no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis.replace('\n','') + '\n')
                fout.flush()
    return 0
                
#Main method as suggested by van Rossum, simplified
def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        opts, args = getopt.getopt(argv[1:], "hm:", ["help", "model"])
    except:
        print ("Error in args : " + str(argv[1:]))
        return 2

    for o in opts:
        if o[0] == "-m":
            config["checkpoint_dir"] = o[1]

    if len(args):
        config["input"] = args[0]
    generate_from_file(config["input"])

if __name__ == "__main__":
    sys.exit(main())
