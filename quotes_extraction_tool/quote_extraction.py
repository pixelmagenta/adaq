import argparse
import spacy
from spacy.tokens import Token, DocBin, Doc, Span
import spacy_udpipe

import verb_cue_classifier
import content_classifier
import source_classifier
import content_resolver
import source_resolver
import quote_resolver

#initialization of necessary pipelines
def init_pipeline(nlp, text_features=False, ner=False):
    if ner:
        ner_vcc = spacy.load("verb-cue-classifier/output/model-best")
        nlp.add_pipe("ner", source=ner_vcc, name="ner_vcc", before="ner")
    nlp.add_pipe('verb_cue_classifier')
    nlp.add_pipe('content_classifier_features')
    if text_features:
        nlp.add_pipe('content_classifier_text_features')
        nlp.add_pipe('content_text_classifier')
    else:
        nlp.add_pipe('content_classifier')
    nlp.add_pipe('source_classifier_features')
    if text_features:
        nlp.add_pipe('source_classifier_text_features')
        nlp.add_pipe('source_text_classifier')
    else:
        nlp.add_pipe('source_classifier')
    nlp.add_pipe('content_resolver')
    nlp.add_pipe('source_resolver')
    nlp.add_pipe('quote_resolver')
    return nlp


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extraction of quotations from text.')
    parser.add_argument('text', type=str, help='text with quotations')
    parser.add_argument('--lang', choices=['en', 'cs', 'ru'], help='languages of text: en, cs, or ru')

    args = parser.parse_args()
    text = args.text

    if args.lang == 'en':
        nlp = init_pipeline(spacy.load("en_core_web_sm"), text_features=True, ner=True)
    elif args.lang == 'ru':
        nlp = init_pipeline(spacy.load("ru_core_news_sm"), text_features=True)
    elif args.lang == 'cs':
        nlp = init_pipeline(spacy_udpipe.load("cs"), text_features=True)

    doc = nlp.make_doc(text)
    doc = nlp(doc)

    print("Detected quotations:")
    for q in doc._.quotes.values():
        print(q)