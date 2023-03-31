from spacy.tokens import Token, Doc
from spacy.language import Language
from spacy.training import iob_to_biluo, biluo_tags_to_spans
import pickle

if not Token.has_extension("content_features"):
    Token.set_extension("content_features", default=[])
    
@Language.component("content_classifier_text_features",
                   assigns=["token._.content_features"],
                   requires=["doc._.verb_cues"])         
def content_classifier_lang_features(doc):
    for token in doc:
        token._.content_features['text'] = token.text
        token._.content_features['lemma'] = token.lemma_
        token._.content_features['prev_5_text'] = doc[max(0, token.i - 5):token.i].text
        token._.content_features['next_5_text'] = doc[token.i+1:token.i+6].text
    return doc

@Language.component("content_classifier_features",
                   assigns=["token._.content_features"],
                   requires=["doc._.verb_cues"])         
def content_classifier_features(doc):
    for token in doc:
        token._.content_features = {
            'spacy_pos': token.pos_, #token.pos,
            'spacy_tag': token.tag_, #token.tag,
            'spacy_iob': token.ent_iob_, #token.ent_iob,
            # odd number of quotation marks to the left means that the token is inside quotation marks
            'in_quote': doc[:token.i].text.count('"') % 2 != 0,
            'dependency_depth': sum(1 for _ in token.ancestors),
            'dependency_relation': token.dep_, #token.dep,
            'verb_cue_child': False,  # default value that is going to be rewritten
            'verb_cue_leftmost_child': False,  # default value that is going to be rewritten
            'follows_verb_cue': False,  # default value that is going to be rewritten
            'sentence_has_verb_cue': False,  # default value that is going to be rewritten
            'index': token.i,
            # custom features
            'is_quote': token.text == '"'
        }
    for verb_cue_span in doc._.verb_cues:
        verb_cue = verb_cue_span.root
        for token in verb_cue.children:
            token._.content_features['verb_cue_child'] = True
        for token in verb_cue.lefts:
            token._.content_features['verb_cue_leftmost_child'] = True
            break
        if verb_cue.i+1 < len(doc):
            doc[verb_cue.i + 1]._.content_features['follows_verb_cue'] = True
        for token in verb_cue.sent:
            token._.content_features['sentence_has_verb_cue'] = True
    return doc


@Language.factory("content_text_classifier",
                   assigns=["token._.content_label", "doc._.content_spans"],
                   requires=["token._.content_features"])
def create_qcc(nlp, name):
    return ContentClassifier('qcc_model.pkl') 

@Language.factory("content_classifier",
                   assigns=["token._.content_label", "doc._.content_spans"],
                   requires=["token._.content_features"])
def create_nolang_qcc(nlp, name):
    return ContentClassifier('no_lang_qcc.pkl') 


class ContentClassifier:
    
    def __init__(self, path):
        if not Token.has_extension("content_label"):
            Token.set_extension("content_label", default='O')

        if not Doc.has_extension("content_spans"):
            Doc.set_extension("content_spans", default=[])

        with open(path, 'rb') as f:
            self.crf = pickle.load(f)
    
    def __call__(self, doc):
        X = []
        for sent in doc.sents:
            X.append([t._.content_features for t in sent])
        y_pred = self.crf.predict(X)

        for sent, pred in zip(doc.sents, y_pred):
            for token, label in zip(sent, pred):
                token._.content_label = label
        doc._.content_spans = biluo_tags_to_spans(
            doc,
            iob_to_biluo([token._.content_label + ("-CONTENT" if token._.content_label != 'O' else '') for token in doc])
        )
        return doc