"""
Predicts if source and cue belong to the same quotation
"""

from parc3corpus import Parc3Corpus
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from spacy.tokens import Token, Doc
from collections import defaultdict
from spacy.language import Language
import spacy
import tqdm
import pickle

def get_source_resolver_features(verb_cue, source):
    dist = min(abs(source.start - verb_cue.i),
               abs(source.end - verb_cue.i))
    same_sent = verb_cue.sent in source.sents
    same_parenthetical = verb_cue.doc[:verb_cue.i].text.count(',') == verb_cue.doc[:source.start].text.count(',')
    return dist, same_sent, same_parenthetical

def create_training_set(examples, nlp):
    X = []
    y = []
    for example in tqdm.tqdm(examples):
        doc = nlp(example.reference)
        for verb_cue in doc._.verb_cues:
            for source in doc._.source_spans:
                X.append(get_source_resolver_features(verb_cue.root, source))
                y.append(1 if verb_cue.label == source.label else 0)
    return X, y

#pipeline for the source resolver model
@Language.factory("source_resolver",
                   assigns=["doc._.cue_to_source"],
                   requires=["doc._.verb_cues", "doc._.source_spans"])
def create_source_resolver(nlp, name):
    return SourceResolver('source_resolver_model.pkl') 


class SourceResolver:
    
    def __init__(self, path):
        if not Doc.has_extension("cue_to_source"):
            Doc.set_extension("cue_to_source", default=defaultdict(list))

        with open(path, 'rb') as f:
            self.clf = pickle.load(f)
    
    def __call__(self, doc):
        for verb_cue in doc._.verb_cues:
            for source in doc._.source_spans:
                match = self.clf.predict([get_source_resolver_features(verb_cue.root, source)])
                if match == 1:
                    doc._.cue_to_source[verb_cue].append(source)
        return doc

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    #ner_vcc = spacy.load("verb-cue-classifier/output/model-best")
    #nlp.add_pipe("ner", source=ner_vcc, name="ner_vcc", before="ner")
    #nlp.add_pipe('verb_cue_classifier')
    #nlp.add_pipe('content_classifier_features')

    print('parsing training set')
    c = Parc3Corpus(f'./data/parc3/train/')
    X_train, y_train = create_training_set(c(nlp), nlp)
    print('parsing test set')
    c = Parc3Corpus(f'./data/parc3/test/')
    X_test, y_test = create_training_set(c(nlp), nlp)
    print('parsing dev set')
    c = Parc3Corpus(f'./data/parc3/dev/')
    X_dev, y_dev = create_training_set(c(nlp), nlp)

    #training of the source resolver model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('F1 on test:', classification_report(y_test, y_pred))

    with open('source_resolver_model.pkl', 'wb') as f:
        pickle.dump(clf, f)