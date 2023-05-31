Quotes Extraction Tool is a program to extract quotations from text in English, Czech and Russian.

This folder contains the model that showed best performance (ML-based model trained on all features) and all necessary modules.

Install required packages:
`pip install requirements.txt`

It is also nesessary to install language models, if they are not installed yet:
`python -m spacy download en_core_web_sm`
`python -m spacy download ru_core_news_sm`

Usage: 
`python quote_extraction.py 'text' --lang ['en', 'cs', 'ru']`