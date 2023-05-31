#!/bin/bash
python3 -m spacy train config.cfg --paths.train ../data/parc3/train.spacy --paths.dev ../data/parc3/dev.spacy --gpu-id 0 --output ./output