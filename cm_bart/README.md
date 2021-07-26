## Code for generating metaphors using the CM-BART model.

## Installation
This project requires an augmented version of Fairseq, included in the `./fairseq/` directory.

To install, navigate to the `./fairseq/` directory, use the following:

`pip install --editable ./`

Note that the using the original fairseq intallation will not work: we encourage you to set up a virtual environment and install the necessary requirements there, to ensure compatability.

## Training the model
To train the model, we first require the corpus data. This can be obtained 


## Generating metaphors
To generate metaphors, we require the model checkpoint. This can be downloaded (for now) from the following Google Drive link:
 
 
 use the `Generate.py` script.

```
python Generate.py [-m model_checkpoint] input_path
```

The input_path should contain one sentence per line, properly formatted as in the paper. See `./data/test/knownmappings.input` for an example.
