from spacy.tokens import Token, Doc
from spacy.language import Language
from spacy.training import iob_to_biluo, biluo_tags_to_spans, biluo_to_iob, offsets_to_biluo_tags
import pickle

if not Token.has_extension("source_features"):
    Token.set_extension("source_features", default=[])
    
@Language.component("source_classifier_text_features",
                   assigns=["token._.source_features"],
                   requires=["doc._.verb_cues"])         
def source_classifier_lang_features(doc):
    for token in doc:
        token._.source_features['text'] = token.text
        token._.source_features['lemma'] = token.lemma_
        token._.source_features['prev_5_text'] = doc[max(0, token.i - 5):token.i].text
        token._.source_features['next_5_text'] = doc[token.i+1:token.i+6].text
    return doc


@Language.component("source_classifier_features",
                   assigns=["token._.source_features"],
                   requires=["doc._.verb_cues", "token._.content_label"])         
def source_classifier_features(doc):
    content_labels = offsets_to_biluo_tags(doc, [(span.start_char, span.end_char, 'content')
                                        for span in doc._.content_spans])
    content_labels = biluo_to_iob(content_labels)
    for content_label, token in zip(content_labels, doc):
        token._.source_features = {
            'spacy_pos': token.pos_, #token.pos,
            'spacy_tag': token.tag_, #token.tag,
            'spacy_ent_iob': token.ent_iob_, #token.ent_iob,
            # odd number of quotation marks to the left means that the token is inside quotation marks
            'in_quote': doc[:token.i].text.count('"') % 2 != 0,
            'dependency_depth': sum(1 for _ in token.ancestors),
            'dependency_relation': token.dep_, #token.dep,
            'verb_cue_child': False,  # default value that is going to be rewritten
            'verb_cue_leftmost_child': False,  # default value that is going to be rewritten
            'follows_verb_cue': False,  # default value that is going to be rewritten
            'sentence_has_verb_cue': False,  # default value that is going to be rewritten
            'index': token.i,
            'content_classifier_label': content_label,
            'spacy_ent_type': token.ent_type_,
            'verb_cue_rightmost_child': False,
            # custom features
            'is_quote': token.text == '"'
        }
    for verb_cue_span in doc._.verb_cues:
        verb_cue = verb_cue_span.root
        for token in verb_cue.children:
            token._.source_features['verb_cue_child'] = True
        for token in verb_cue.subtree:
            if token != verb_cue:
                token._.source_features['verb_cue_distance'] = abs(token.i - verb_cue.i)
        for token in verb_cue.lefts:
            token._.source_features['verb_cue_leftmost_child'] = True
            break
        rightmost_token = None
        for rightmost_token in verb_cue.rights: pass
        if rightmost_token is not None:
            rightmost_token._.source_features['verb_cue_rightmost_child'] = True
        if verb_cue.i+1 < len(doc):
            doc[verb_cue.i + 1]._.source_features['follows_verb_cue'] = True
        for token in verb_cue.sent:
            token._.source_features['sentence_has_verb_cue'] = True
    return doc


@Language.factory("source_text_classifier",
                   assigns=["token._.source_label"],
                   requires=["token._.source_features"])
def create_qsc(nlp, name):
    return SourceClassifier('qsc_model.pkl') 

@Language.factory("source_classifier",
                   assigns=["token._.source_label"],
                   requires=["token._.source_features"])
def create_nolang_qsc(nlp, name):
    return SourceClassifier('no_lang_qsc.pkl') 

class SourceClassifier:
    
    def __init__(self, path):
        if not Token.has_extension("source_label"):
            Token.set_extension("source_label", default='O')

        if not Doc.has_extension("source_spans"):
            Doc.set_extension("source_spans", default=[])
        with open(path, 'rb') as f:
            self.crf = pickle.load(f)
    
    def __call__(self, doc):
        X = []
        for sent in doc.sents:
            X.append([t._.source_features for t in sent])
        y_pred = self.crf.predict(X)
        for sent, pred in zip(doc.sents, y_pred):
            for token, label in zip(sent, pred):
                token._.source_label = label
        
        doc._.source_spans = biluo_tags_to_spans(
            doc,
            iob_to_biluo([token._.source_label + ("-SOURCE" if token._.source_label != 'O' else '') for token in doc])
        )
        return doc