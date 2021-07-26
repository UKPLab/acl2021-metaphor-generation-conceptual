## Code for generating metaphors using the CM-BART model.

## Installation
This project requires an augmented version of Fairseq, included in the `./fairseq/` directory.

To install, navigate to the `./fairseq/` directory, use the following:

`pip install --editable ./`

Note that the using the original fairseq intallation will not work: we encourage you to set up a virtual environment and install the necessary requirements there, to ensure compatability.

## Generating metaphors
To generate metaphors, we require the model checkpoint. This can be downloaded (for now) from the following Google Drive link:
 
 
 use the `Generate.py` script.

```
python Generate.py [-m model_checkpoint] input_path
```

The input_path should contain one sentence per line, properly formatted as in the paper. See `./data/test/knownmappings.input` for an example.


## Fine-tuning the model
To fine-tune the model, we first require the corpus data. This can be obtained (for now) via  the following Google Drive folder:

https://drive.google.com/drive/folders/138SCh3xANO4hgs0IAId5M5lMgLrKUTPk?usp=sharing

Download the files to the `./data/` directory. Note the file names use "source" and "target" in the MT sense, NOT the conceptual metaphor sense.

We also require the initial BART model. This can be acquired the fairseq repository (we use `bart.large`):
https://github.com/pytorch/fairseq/blob/master/examples/bart/README.md

Download a model to the `./models/` directory.

Finally, we execute the `finetune.sh` script to fine-tune the model on the provided data.

Make sure to update the `BART_PATH` variable to your local BART model. 