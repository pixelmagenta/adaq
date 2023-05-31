"""
Loads and preprocesses PARC dataset
"""

from typing import Optional, Union, List, Iterable, Iterator, TYPE_CHECKING, Callable
from pathlib import Path
from spacy import util
from spacy.tokens import Token, Doc, Span
from spacy.training.corpus import walk_corpus
from spacy.training import Example
from lxml import etree
from collections import defaultdict
import spacy

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

class Parc3Corpus:
    file_type = "xml"

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
        for loc in walk_corpus(self.path, ".xml"):
            doc = self._load_xml(loc, nlp)
            if self.min_length >= 1 and len(doc) < self.min_length:
                continue
            elif self.max_length >= 1 and len(doc) >= self.max_length:
                continue
            else:
                yield Example(nlp.make_doc(doc.text), doc)
    
    def _load_xml(self, path, nlp):
        parser = etree.XMLParser(resolve_entities=False)
        with open(path) as f:
            tree = etree.parse(f, parser)

        words, spaces = [], []
        sent_starts = []
        prev_end = None
        word_attributions = []
        
        # there must not be two cue annotations
        cue_conflicts = set()
        
        for sent in tree.iter('SENTENCE'):
            sent_start = True
            for word in sent.iter('WORD'):
                byte_start, byte_end = map(int, word.get("ByteCount").split(','))
                words.append(self._preprocess_word(word.get('text')))
                if prev_end is not None:
                    spaces.append(byte_start != prev_end)
                prev_end = byte_end
                sent_starts.append(sent_start)
                sent_start = False
                
                attributions = []
                prev_cue_attr = None
                
                for attr in word.iter('attribution'):
                    attr_id = attr.get("id")
                    if 'Nested_relation' in attr_id:
                        # we skip nested relations
                        continue
                    for role in attr.iter('attributionRole'):
                        attr_role = role.get("roleValue")
                    attributions.append((attr_id, attr_role))
                word_attributions.append(attributions)
        
        doc = Doc(nlp.vocab,
                  words,
                  spaces + [False],
                  sent_starts=sent_starts)
        quotes = defaultdict(lambda: {'source': [], 'cue': [], 'content': []})
        for token, attributions in zip(doc, word_attributions):
            for attr_id, attr_role in attributions:
                if len(quotes[attr_id][attr_role]) == 0 or quotes[attr_id][attr_role][-1][1] != token.i:
                    quotes[attr_id][attr_role].append((token.i, token.i + 1))
                else:
                    tmp = quotes[attr_id][attr_role][-1]
                    quotes[attr_id][attr_role][-1] = (tmp[0], token.i+1)
            #token._.attributions += list(filter(lambda x: x[0] not in cue_conflicts, attributions))
        for attr_id, attr in quotes.items():
            attributions = {}
            for attr_type in ['cue', 'content', 'source']:
                attributions[attr_type] = [Span(doc, start, end, label=attr_id)
                                           for start, end in attr[attr_type]]
            doc._.attributions[attr_id] = attributions
            doc._.verb_cues += attributions['cue']
            doc._.content_spans += attributions['content']
            doc._.source_spans += attributions['source']
        doc._.verb_cues = util.filter_spans(doc._.verb_cues)
        doc._.content_spans = util.filter_spans(doc._.content_spans)
        doc._.source_spans = util.filter_spans(doc._.source_spans)
        
        for attributions in doc._.attributions.values():
            for cue in attributions['cue']:
                if cue in doc._.verb_cues:
                    for content in attributions['content']:
                        if content in doc._.content_spans:
                            doc._.cue_to_content[cue].append(content)
                    for source in attributions['source']:
                        if source in doc._.source_spans:
                            doc._.cue_to_source[cue].append(source)
        
        doc._.path = path
        
        return doc
    
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