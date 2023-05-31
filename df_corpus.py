"""
Loads Czech and Russian datasets
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
import pandas as pd
import spacy_udpipe


if not Token.has_extension("attributions"):
    Token.set_extension("attributions", default=[])

if not Doc.has_extension("quotes"):
    Doc.set_extension("quotes", default={})
    
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

def update_span_collection(idx, arr):
    if len(arr) == 0 or arr[-1][1] != idx:
        arr.append((idx, idx + 1))
    else:
        arr[-1] = (arr[-1][0], idx + 1)
    return arr

class CsvCorpus:

    def __init__(
        self,
        path_tokens: Optional[Union[str, Path]],
        path_sentences,
        *,
        limit: int = 0,
        min_length: int = 0,
        max_length: int = 0,
    ) -> None:
        self.path_tokens = path_tokens
        self.path_sentences = path_sentences
        self.min_length = min_length
        self.max_length = max_length
        self.limit = limit

    def __call__(self, nlp: "Language") -> Iterator[Example]:
        """Yield examples from the data.
        nlp (Language): The current nlp object.
        YIELDS (Example): The example objects.
        DOCS: https://spacy.io/api/corpus#jsonlcorpus-call
        """
        
        df = pd.read_csv(self.path_tokens, sep='\t')
        df_sentences = pd.read_csv(self.path_sentences, sep='\t')
        #df["tags"] = ['content' if (x == '"' and y == "cue") else y for x, y in zip(df["text"], df["tags"])]
        
        words, spaces, labels = self._spaces_list_construction(df, df_sentences)
        for w, s, l in zip(words, spaces, labels):
            doc = self._load_doc(w, s, l, nlp)
            yield Example(nlp.make_doc(doc.text), doc)


    def _spaces_list_construction(self, df, df_sentences):
        words_lists = []
        spaces_lists = []
        labels_lists = []
        current_sentence_num = ''
        spaces_list = []
        full_sentence = ''
        words_list = []
        labels_list = []
        for index, row in df.iterrows():
            if row["sentence_num"] != current_sentence_num:   
                full_sentence = list(df_sentences[df_sentences["sentence_id"] == row["sentence_num"]]["sentences"])[0]
                words_list = list(df[df["sentence_num"] == row["sentence_num"]]["text"])
                words_lists.append(words_list)
                labels_list = list(df[df["sentence_num"] == row["sentence_num"]]["tags"])
                labels_lists.append(labels_list)
                current_sentence_num = row["sentence_num"]
                if len(spaces_list) > 0:
                    spaces_lists.append(spaces_list)
                spaces_list = []

            full_sentence = full_sentence[len(row["text"]):]
            if len(full_sentence) < 1:
                spaces_list.append("False")
                continue
            else:
                if full_sentence[0] == ' ':
                    spaces_list.append(True)
                    full_sentence = full_sentence[1:]
                else:
                    spaces_list.append(False)

        spaces_lists.append(spaces_list)
        return words_lists, spaces_lists, labels_lists
    
    def _load_doc(self, words, spaces, labels, nlp):
        
        doc = Doc(nlp.vocab,
                  words,
                  spaces,
                  sent_starts=[True] + [False]*(len(words)-1))
        quotes = defaultdict(lambda: {'source': [], 'cue': [], 'content': []})
        for token, label in zip(doc, labels):
            if label != 'None':
                quotes[0][label].append(token)
                token._.attributions.append((0, label))
            if label == 'cue':
                update_span_collection(token.i, doc._.verb_cues)
            elif label == 'content':
                update_span_collection(token.i, doc._.content_spans)
            elif label == 'source':
                update_span_collection(token.i, doc._.source_spans)
        doc._.verb_cues = [Span(doc, start, end)
                           for start, end in doc._.verb_cues]
        doc._.content_spans = [Span(doc, start, end)
                           for start, end in doc._.content_spans]
        doc._.source_spans = [Span(doc, start, end)
                           for start, end in doc._.source_spans]
        for cue in doc._.verb_cues:
            for content in doc._.content_spans:
                doc._.cue_to_content[cue].append(content)
            for source in doc._.source_spans:
                doc._.cue_to_source[cue].append(source)
        doc._.quotes = quotes
        
        return doc