from parc3corpus import Parc3Corpus
from spacy.tokens import DocBin, Doc, Span
from spacy.training import Example
import spacy

nlp = spacy.load("en_core_web_sm")

for training_set in ['dev', 'test', 'train']:
    c = Parc3Corpus(f'./data/parc3/{training_set}/')
    db = DocBin()
    for ex in c(nlp):
        doc = ex.reference
        doc.set_ents([Span(doc, s.start, s.end, label='VERBCUE') for s in doc._.verb_cues])
        db.add(doc)
    db.to_disk(f"./data/parc3/{training_set}.spacy")