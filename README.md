This is the repository for my Master thesis research: **"Automatic Detection and Attribution of Quotes"**.

### Abstract

Quotations extraction and attribution are important practical tasks for the media, but most of the presented solutions are monolingual. In this work, I present a complex machine learning-based system for extraction and attribution of direct and indirect quotations, which is trained on English and tested on Czech and Russian data. Czech and Russian test datasets were manually annotated as part of this study. This system is compared against a rule-based baseline model. Baseline model demonstrates better precision in extraction of quotation elements, but low recall. The machine learning-based model is better overall in extracting separate elements of quotations and full quotations as well.

### Description

The setup for the experiments consists of 13 Python files. I created separate Python files for different components of the systems. These are files for loading and preprocessing of the datasets, files for generating models' features, and converting the training results into the needed form. Additionally, there are files for running the training of the models. These files can be run only for a complete recreation of the workflow. The trained models are available as `.pkl` files.

The verb cue classifier was trained separately using SpaCy's framework, and it is stored in the folder with the same name. It is also a required component for the experiments.

The repository contains the folder `quotes_extraction_tool` with the files for the tool and instructions how to use it.

The folder `data` comprises annotated Czech and Russian files. For each language, there are two files: one is with complete sentences, and the other one is with annotated tokens.

### Requirements

A full list of required packages is in the file `requirements.txt`. Additionally, it is required to have installed SpaCy language packages `en core web sm` and `ru core news sm`. It can be done with the following commands:

```
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm
```

To evaluate the SiR dataset, the pybrat library must be modified since it does not support the relation annotations used in SiR. I provide the modification in the folder pybrat, and it can be installed with the following command:

```
cd pybrat && python setup.py install
```

### Quotes Extraction Experiments

All experiments were conducted in Jupyter notebooks in the following files:

* `baseline experiments.ipynb` – evaluation of baseline model
* `ML-based system experiments.ipynb` – evaluation of ML-based systems
* `sir evaluation.ipynb` – evaluation of SiR dataset.