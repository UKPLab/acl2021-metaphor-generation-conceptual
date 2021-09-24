### CM Lex
Generating from conceptual metaphors using lexical mappings.


## Training embeddings
   
Use the `EMtrainer.py` script, which takes relevant arguments for embedding training.

Note that to train embeddings, the training corpus is needed. It is not provided here due to size restrictions: please instead find (for now) at the following Google drive link:

https://drive.google.com/file/d/1YJ9IwBOeLL8TlK-VvROGAm46kLafYZPg/view?usp=sharing

Place this file in the `./data/` directory.

```
python EMTrainer.py -i: input path (the corpus provided at the link above)
                    -o: output path
                    -m: model type
                    -d: dimensionality
```

Model type can be w2v, ft (fasttext) or glove.

Our best model used 50-dimension w2v vectors:

```
python EMTrainer.py -i ../data/frame_corpus.txt -o ../models/w2v_50/ -m w2v -d 50
```

## Generating mappings
The mapping file between frames is not provided due to size, please instead find (for now) at the following Google Drive link:

https://drive.google.com/file/d/1Db5W4aN52syuwMJsNkT029a7ZB3rl4Ks/view?usp=sharing

Place this file in the `./mappings/` directory.

Or, generate them from a model! Run the `Mapping.py` script, providing options for the model used as needed:

```
python Mappings.py -m [model_type] -i [model_file] -o [output_file]
```

By default it will used the provided w2v model, and store the mappings in the mappings directory.

## Running lexical replacement
Use the  `Generate.py` script. It takes as input a file of test instances.
Test instances require the input sentence, the focus word id, and the input (target) frame. See `tests/final.tsv` for the standard formatting, or provide your own data directly to the `generate` function of `LexicalReplacement.py`.

You can specify which mapping and embedding model to use with the flags `-m` and `-e`. The system defaults to the provided models, which were used for the paper.

```
python Generate.py -m mappings/w2v_50.json -e models/w2v_50 [input_file.tsv]
```

To run generation on the test data from the paper, use the `ModelTesting.py` script:

```
python ModelTesting.py
```
