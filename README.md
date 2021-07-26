This repository is for the paper <a href="https://arxiv.org/pdf/2106.01228.pdf">Metaphor Generation with Conceptual Mappings</a>.

## System Demonstration
Our <a href="http://metaphorgenerator.cs.columbia.edu:8080/">system demonstration</a> based on the CM-BART model is now available, try it out!

## Models
This paper defines two models for metaphor generation based on conceptual mappings: <a href="https://github.com/UKPLab/acl2021-metaphor-generation-conceptual/tree/main/cm_lex">CM-Lex</a> and <a href="https://github.com/UKPLab/acl2021-metaphor-generation-conceptual/tree/main/cm_bart">CM-BART</a>. CM-Lex uses no training data, and works at the word level, while CM-BART is a BART-based seq2seq model.

CM-BART will generally yield better performance, and doesn't require knowing which word to change. If you know which word needs to be changed, CM-Lex may be appropriate.

These models each have their own directory and requirements. See their respective READMEs for more details.
