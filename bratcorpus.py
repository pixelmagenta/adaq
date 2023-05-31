"""
Reads SiR dataset and converts to the format compatible with the experiments code
"""

from typing import Optional, Union, List, Iterable, Iterator, TYPE_CHECKING, Callable
from pathlib import Path
from spacy import util
from spacy.tokens import Token, Doc, Span
from spacy.training.corpus import walk_corpus
from spacy.training import Example
from collections import defaultdict
import spacy
import copy
from pybrat import BratAnnotations, BratText

from difflib import SequenceMatcher

if not Token.has_extension("attribution"):
    Token.set_extension("attribution", default=None)

if not Doc.has_extension("attributions"):
    Doc.set_extension("attributions", default={})
    
if not Doc.has_extension("verb_cues"):
    Doc.set_extension("verb_cues", default=[])
    
if not Doc.has_extension("content_spans"):
    Doc.set_extension("content_spans", default=[])
    
if not Doc.has_extension("source_spans"):
    Doc.set_extension("source_spans", default=[])
    
if not Doc.has_extension("cue_to_content"):
    Doc.set_extension("cue_to_content", default=defaultdict(list))

if not Doc.has_extension("cue_to_source"):
    Doc.set_extension("cue_to_source", default=defaultdict(list))
    
if not Doc.has_extension("path"):
    Doc.set_extension("path", default='')

class BratCorpus:
    file_type = "ann"

    def __init__(
        self,
        path: Optional[Union[str, Path]],
        *,
        limit: int = 0,
        min_length: int = 0,
        max_length: int = 0,
    ) -> None:
        self.path = util.ensure_path(path)
        self.min_length = min_length
        self.max_length = max_length
        self.limit = limit

    def __call__(self, nlp: "Language") -> Iterator[Example]:
        """Yield examples from the data.
        nlp (Language): The current nlp object.
        YIELDS (Example): The example objects.
        DOCS: https://spacy.io/api/corpus#jsonlcorpus-call
        """
        for loc in walk_corpus(self.path, ".ann"):
            orig, doc = self._load_brat(str(loc), nlp)
            if self.min_length >= 1 and len(doc) < self.min_length:
                continue
            elif self.max_length >= 1 and len(doc) >= self.max_length:
                continue
            else:
                yield Example(orig, doc)
    
    def _load_brat(self, path, nlp):
        anns = BratAnnotations.from_file(path)
        anntxt = BratText.from_files(text=path.rstrip('.ann') + '.txt')
        
        doc = nlp.make_doc(anntxt.text())
        
        spans = {}
        doc_span = None
        qwe = []
        chr_map = []
        doc_idx = 0
        
        gold_text, spacy_text = anntxt.text(), doc.text
        s = SequenceMatcher(lambda x: x == "\n", gold_text, spacy_text)
        
        chr_map = list(range(len(gold_text)))
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == 'equal':
                for x, y in zip(range(i1, i2), range(j1, j2)):
                    chr_map[x] = y
            elif tag == 'replace':
                pass
            elif tag == 'delete':
                for x in range(i1, i2):
                    chr_map[x] = chr_map[i1 - 1]
            elif tag == 'insert':
                pass
            
        for span in anns.spans:
            if span.type == 'PHRASE' or span.type == 'PHRASE1':
                doc_span = doc.char_span(chr_map[span.start_index], chr_map[span.end_index],
                                         alignment_mode='expand', label="cue")
                doc._.verb_cues.append(doc_span)
            elif 'PHRASE' in span.type:
                # skip if annotation was not agreed upon.
                continue
            elif not span.type[-1].isdigit() or span.type[-1] == '1':
                doc_span = doc.char_span(chr_map[span.start_index], chr_map[span.end_index],
                                         alignment_mode='expand', label="source")
                doc._.source_spans.append(doc_span)
            if span.text != doc_span.text:
                print('WARN: Mismatch', path, doc_span, '|', span)
            spans[span.id] = doc_span
    
        for rel in anns._relations:
            a, b = rel['ref_spans']
            span_a = spans.get(a[1], None)
            span_b = spans.get(b[1], None)
            if span_a is None or span_b is None:
                continue
            if span_a.label == 'cue':
                cue, source = span_a, span_b
            else:
                cue, source = span_b, span_a
            doc._.cue_to_source[cue].append(source)
        
        doc._.path = path
        
        return nlp.make_doc(anntxt.text()), doc
    
    def _preprocess_word(self, word):
        if word == '``' or word == "''":
            return '"'
        if word == '-LRB-':
            return "("
        if word == '-LCB-':
            return "{"
        if word == '-RRB-':
            return ")"
        if word == '-RCB-':
            return "}"
        return word.replace('\\', '')