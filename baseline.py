"""
Runs baseline model
"""

from cytoolz import itertoolz
from spacy.symbols import (
    AUX, VERB,
    agent, attr, aux, auxpass, csubj, csubjpass, dobj, neg, nsubj, nsubjpass, obj, pobj, xcomp,
)
from operator import attrgetter
from spacy.tokens import Token, Doc, Span
from spacy.language import Language
from spacy.lang.en import English
from spacy.lang.ru import Russian
from spacy.lang.cs import Czech
from spacy import util

#pipeline for baseline model for Czech
@Czech.factory("rule_based_attribution")
def create_cs_rbt(nlp, name):
    return RuleBasedAttribution({'apelovat', 'avizovat', 'charakterizovat', 'deklarovat', 'dodat', 'dodávat', 'doplnit', 'hodnotit',
'hovořit', 'informovat', 'komentovat', 'konstatovat', 'kvitovat', 'líčit', 'namítat', 'namítnout',
'napsat', 'objasnit', 'odmítnout', 'odpovědět', 'odvětit', 'okomentovat', 'oznámit', 'popsat',
'potvrdit', 'potvrzovat', 'poukázat', 'poznamenat', 'přiznat se', 'prohlásit', 'proklamovat',
'prozradit', 'reagovat', 'říci', 'říkat', 'sdělit', 'tvrdit', 'upozornit', 'upozorňovat', 'uvést',
'varovat', 'vyjádřit se', 'vylíčit', 'vypovědět', 'vyslovit se', 'vysvětlit', 'vysvětlovat', 'vyzpovídat',
'zareagovat', 'zdůraznit', 'zmínit', 'zpovídat'})

#pipeline for baseline model for Russian
@Russian.factory("rule_based_attribution")
def create_ru_rbt(nlp, name):
    return RuleBasedAttribution({"сказать",
        "заявить",
        "передать",
        "объявить",
       "подчеркнуть",'призвать', 'объявить', 'характеризовать', 'декларировать', 'дополнить', 'оценить', 'говорить', 'информировать', 'заявлять', 'признавать', 'признать', 'изображать', 'спорить','возражать', 'писать', 'уточнять', 'отклонять', 'отвечать', 'комментировать', 'сообщать', 'описывать',
'подтверждать', 'утверждать', 'указывать', 'замечать', 'признаваться', 'провозглашать',
'раскрывать', 'рассказывать', 'утверждать', 'указывать', 'указать',
'предупреждать', 'выражать', 'изобразить', 'свидетельствовать', 'произносить', 'объяснять', 'излагать', 'разъяснять', 'исповедовать',
'реагировать', 'подчеркивать', 'упоминать', 'признаваться'})

#pipeline for baseline model for English
@English.factory("rule_based_attribution")
def create_en_rbt(nlp, name):
    return RuleBasedAttribution({'accord',
 'accuse',
 'acknowledge',
 'add',
 'admit',
 'agree',
 'allege',
 'announce',
 'argue',
 'ask',
 'assert',
 'believe',
 'blame',
 'charge',
 'cite',
 'claim',
 'complain',
 'concede',
 'conclude',
 'confirm',
 'contend',
 'criticize',
 'declare',
 'decline',
 'deny',
 'describe',
 'disagree',
 'disclose',
 'estimate',
 'explain',
 'fear',
 'hope',
 'insist',
 'maintain',
 'mention',
 'note',
 'observe',
 'order',
 'predict',
 'promise',
 'recall',
 'recommend',
 'reply',
 'report',
 'say',
 'state',
 'stress',
 'suggest',
 'tell',
 'testify',
 'think',
 'urge',
 'warn',
 'worry',
 'write'})

_NOMINAL_SUBJ_DEPS = {nsubj, nsubjpass}
_CLAUSAL_SUBJ_DEPS = {csubj, csubjpass}
_ACTIVE_SUBJ_DEPS = {csubj, nsubj}
_VERB_MODIFIER_DEPS = {aux, auxpass, neg}


class RuleBasedAttribution:
    def __init__(self, verbs):
        self.reporting_verbs = verbs
        if not Token.has_extension("attributions"):
            Token.set_extension("attributions", default=[])
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

    def __call__(self, doc):
        qtok_idxs = [tok.i for tok in doc if tok.text == '"' and tok.pos_ == 'PUNCT']
        
        if len(qtok_idxs) % 2 != 0:
            print(
                f"{len(qtok_idxs)} quotation marks found, indicating an unclosed quotation; "
                "given the limitations of this method, it's safest to bail out "
                "rather than guess which quotation is unclosed"
            )
            return doc
        qtok_pair_idxs = list(itertoolz.partition(2, qtok_idxs))
        quote_idx = 0
        for qtok_start_idx, qtok_end_idx in qtok_pair_idxs:
            content = doc[qtok_start_idx : qtok_end_idx + 1]
            cue = None
            speaker = None
            # filter quotations by content
            if (
                # quotations should have at least a couple tokens
                # excluding the first/last quotation mark tokens
                len(content) < 4
                # filter out titles of books and such, if possible
                or all(
                    tok.is_title
                    for tok in content
                    # if tok.pos in {NOUN, PROPN}
                    if not (tok.is_punct or tok.is_stop)
                )
                # TODO: require closing punctuation before the quotation mark?
                # content[-2].is_punct is False
            ):
                continue
            # get window of adjacent/overlapping sentences
            window_sents = (
                sent
                for sent in doc.sents
                # these boundary cases are a subtle bit of work...
                if (
                    (sent.start < qtok_start_idx and sent.end >= qtok_start_idx - 1)
                    or (sent.start <= qtok_end_idx + 1 and sent.end > qtok_end_idx)
                )
            )
            # get candidate cue verbs in window
            cue_cands = [
                tok
                for sent in window_sents
                for tok in sent
                if (
                    tok.pos == VERB
                    and tok.lemma_ in self.reporting_verbs
                    # cue verbs must occur *outside* any quotation content
                    and not any(
                        qts_idx <= tok.i <= qte_idx
                        for qts_idx, qte_idx in qtok_pair_idxs
                    )
                )
            ]
            # sort candidates by proximity to quote content
            cue_cands = sorted(
                cue_cands,
                key=lambda cc: min(abs(cc.i - qtok_start_idx), abs(cc.i - qtok_end_idx)),
            )
            for cue_cand in cue_cands:
                if cue is not None:
                    break
                for speaker_cand in cue_cand.children:
                    if speaker_cand.dep in _ACTIVE_SUBJ_DEPS:
                        cue = expand_verb(cue_cand)
                        speaker = expand_noun(speaker_cand)
                        break
            verb_cues = token_list_to_spans(doc, cue)
            source_spans = token_list_to_spans(doc, speaker)
            content_spans = token_list_to_spans(doc, content)
            doc._.verb_cues += verb_cues
            doc._.source_spans += source_spans
            doc._.content_spans += content_spans
            if content and cue:
                for verb_cue in verb_cues:
                    for content_span in content_spans:
                        doc._.cue_to_content[verb_cue].append(content_span)
            
            if content and speaker:
                for verb_cue in verb_cues:
                    for source_span in source_spans:
                        doc._.cue_to_source[verb_cue].append(source_span)
            if content and cue and speaker:
                for token in speaker:
                    token._.attributions.append(('source', quote_idx))
                for token in cue:
                    token._.attributions.append(('cue', quote_idx))
                for token in content:
                    token._.attributions.append(('content', quote_idx))
            quote_idx += 1
        doc._.verb_cues = util.filter_spans(doc._.verb_cues)
        doc._.content_spans = util.filter_spans(doc._.content_spans)
        doc._.source_spans = util.filter_spans(doc._.source_spans)
        return doc

def token_list_to_spans(doc, tokens):
    if not tokens:
        return []
    res = []
    for token in tokens:
        if len(res) == 0 or res[-1][1] != token.i:
            res.append((token.i, token.i + 1))
        else:
            res[-1] = (res[-1][0], token.i+1)
    return [Span(doc, start, end) for start, end in res]
    
def expand_noun(tok: Token):
    """Expand a noun token to include all associated conjunct and compound nouns."""
    tok_and_conjuncts = [tok] + list(tok.conjuncts)
    compounds = [
        child
        for tc in tok_and_conjuncts
        for child in tc.children
        # TODO: why doesn't compound import from spacy.symbols?
        if child.dep_ == "compound"
    ]
    return tok_and_conjuncts + compounds


def expand_verb(tok: Token):
    """Expand a verb token to include all associated auxiliary and negation tokens."""
    verb_modifiers = [
        child for child in tok.children if child.dep in _VERB_MODIFIER_DEPS
    ]
    return [tok] + verb_modifiers