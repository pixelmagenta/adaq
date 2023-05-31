"""
Predicts if content and cue belong to the same quotation
"""

from parc3corpus import Parc3Corpus
from spacy.tokens import Token, Doc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import defaultdict
from spacy.language import Language
import spacy
import tqdm
import pickle

def get_content_resolver_features(verb_cue, content):
    dist = min(abs(content.start - verb_cue.i),
               abs(content.end - verb_cue.i))
    same_sent = verb_cue.sent in content.sents
    is_desc = False
    for token in verb_cue.subtree:
        if token in content:
            is_desc = True
            break
    return dist, same_sent, is_desc

def create_training_set(examples, nlp):
    X = []
    y = []
    for example in tqdm.tqdm(examples):
        doc = nlp(example.reference)
        for verb_cue in doc._.verb_cues:
            for content in doc._.content_spans:
                X.append(get_content_resolver_features(verb_cue.root, content))
                y.append(1 if verb_cue.label == content.label else 0)
    return X, y

#pipeline for the content resolver model
@Language.factory("content_resolver",
                   assigns=["doc._.cue_to_content"],
                   requires=["doc._.verb_cues", "doc._.content_spans"])
def create_content_resolver(nlp, name):
    return ContentResolver('content_resolver_model.pkl') 


class ContentResolver:
    
    def __init__(self, path):
        if not Doc.has_extension("cue_to_content"):
            Doc.set_extension("cue_to_content", default=defaultdict(list))

        with open(path, 'rb') as f:
            self.clf = pickle.load(f)
    
    def __call__(self, doc):
        for verb_cue in doc._.verb_cues:
            for content in doc._.content_spans:
                match = self.clf.predict([get_content_resolver_features(verb_cue.root, content)])
                if match == 1:
                    doc._.cue_to_content[verb_cue].append(content)
        return doc


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")

    print('parsing training set')
    c = Parc3Corpus(f'./data/parc3/train/')
    X_train, y_train = create_training_set(c(nlp), nlp)
    print('parsing test set')
    c = Parc3Corpus(f'./data/parc3/test/')
    X_test, y_test = create_training_set(c(nlp), nlp)
    print('parsing dev set')
    c = Parc3Corpus(f'./data/parc3/dev/')
    X_dev, y_dev = create_training_set(c(nlp), nlp)

    #training of the content resolver model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('F1 on test:', classification_report(y_test, y_pred))

    with open('content_resolver_model.pkl', 'wb') as f:
        pickle.dump(clf, f)