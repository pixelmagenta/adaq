"""
Connects together results of content and source resolvers
"""

from spacy.language import Language
from spacy.tokens import Doc

if not Doc.has_extension("quotes"):
    Doc.set_extension("quotes", default={})

#pipeline for the quote resolver function
@Language.component("quote_resolver",
                   assigns=["doc._.quotes"],
                   requires=["doc._.cue_to_content", "doc._.cue_to_source"])
def quote_resolver(doc):
    quote_idx = 0
    quotes = {}
    for cue, content_spans in doc._.cue_to_content.items():
        for source_spans in doc._.cue_to_source[cue]:
            quotes[quote_idx] = {
                'cue': [cue],
                'content': content_spans,
                'source': source_spans
            }
            quote_idx += 1
    doc._.quotes = quotes
    return doc