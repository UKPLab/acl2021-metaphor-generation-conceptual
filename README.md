# Conceptual Metaphor Generation

This repository is for the paper <a href="https://arxiv.org/pdf/2106.01228.pdf">Metaphor Generation with Conceptual Mappings</a>.

Please use the following citation:

```
@inproceedings{stowe-etal-2021-metaphor,
    title = "Metaphor Generation with Conceptual Mappings",
    author = "Stowe, Kevin  and
      Chakrabarty, Tuhin  and
      Peng, Nanyun  and
      Muresan, Smaranda  and
      Gurevych, Iryna",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.524",
    doi = "10.18653/v1/2021.acl-long.524",
    pages = "6724--6736",
}
```
Contact person: Kevin Stowe, stowe@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something is broken or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## System Demonstration
Our <a href="http://metaphorgenerator.cs.columbia.edu:8080/">system demonstration</a> based on the CM-BART model is now available, try it out!

## Models
This paper defines two models for metaphor generation based on conceptual mappings: <a href="https://github.com/UKPLab/acl2021-metaphor-generation-conceptual/tree/main/cm_lex">CM-Lex</a> and <a href="https://github.com/UKPLab/acl2021-metaphor-generation-conceptual/tree/main/cm_bart">CM-BART</a>. CM-Lex uses no training data, and works at the word level, while CM-BART is a BART-based seq2seq model.

CM-BART will generally yield better performance, and doesn't require knowing which word to change. If you know which word needs to be changed, CM-Lex may be appropriate.

These models each have their own directory and requirements. See their respective READMEs for more details.
