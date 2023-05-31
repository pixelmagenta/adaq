"""
Predicts cue labels
"""

from spacy.tokens import Token, Doc, Span
from spacy.lang.en import English
from spacy.lang.ru import Russian
from spacy.lang.cs import Czech
import spacy

#pipeline for the cue classifier for English
@English.factory("verb_cue_classifier",
                assigns=["doc._.verb_cues"])
def create_en_vcc(nlp, name):
    return VerbCueClassifier()

#pipeline for the cue classifier for Czech
@Czech.factory("verb_cue_classifier",
                assigns=["doc._.verb_cues"])
def create_cz_vcc(nlp, name):
    return VerbCueClassifier({'apelovat', 'avizovat', 'charakterizovat', 'deklarovat', 'dodat', 'dodávat', 'doplnit', 'hodnotit',
'hovořit', 'informovat', 'komentovat', 'konstatovat', 'kvitovat', 'líčit', 'namítat', 'namítnout',
'napsat', 'objasnit', 'odmítnout', 'odpovědět', 'odvětit', 'okomentovat', 'oznámit', 'popsat',
'potvrdit', 'potvrzovat', 'poukázat', 'poznamenat', 'přiznat se', 'prohlásit', 'proklamovat',
'prozradit', 'reagovat', 'říci', 'říkat', 'sdělit', 'tvrdit', 'upozornit', 'upozorňovat', 'uvést',
'varovat', 'vyjádřit se', 'vylíčit', 'vypovědět', 'vyslovit se', 'vysvětlit', 'vysvětlovat', 'vyzpovídat',
'zareagovat', 'zdůraznit', 'zmínit', 'zpovídat'})

#pipeline for the cue classifier for Russian
@Russian.factory("verb_cue_classifier",
                assigns=["doc._.verb_cues"])
def create_ru_vcc(nlp, name):
    return VerbCueClassifier({"сказать",
        "заявить",
        "передать",
        "объявить",
       "подчеркнуть",'призвать', 'объявить', 'характеризовать', 'декларировать', 'дополнить', 'оценить', 'говорить', 'информировать', 'комментировать', 'заявлять', 'признавать', 'признать', 'изображать', 'спорить','возражать', 'писать', 'уточнять', 'отклонять', 'отвечать', 'комментировать', 'сообщать', 'описывать',
'подтверждать', 'утверждать', 'указывать', 'замечать', 'признаваться', 'заявлять', 'провозглашать',
'раскрывать', 'рассказывать', 'утверждать', 'указывать', 'указать',
'предупреждать', 'выражать', 'изобразить', 'свидетельствовать', 'произносить', 'объяснять', 'излагать', 'разъяснять', 'исповедовать',
'реагировать', 'подчеркивать', 'упоминать', 'признаваться'})


class VerbCueClassifier:
    def __init__(self, verbs=None):
        if not Doc.has_extension("verb_cues"):
            Doc.set_extension("verb_cues", default=[])
        if verbs is None:
            self.verbs = set()
        else:
            self.verbs = verbs

    def __call__(self, doc):
        #doc = self.model(doc)
        verb_cues = []
        for token in doc:
            if token.ent_type_ == 'VERBCUE' or token.lemma_ in self.verbs:
                if len(verb_cues) == 0 or verb_cues[-1][1] != token.i:
                    verb_cues.append((token.i, token.i + 1))
                else:
                    tmp = verb_cues[-1]
                    verb_cues[-1] = (tmp[0], token.i+1)
        doc._.verb_cues = [Span(doc, start, end, label='verb_cue')
                                           for start, end in verb_cues]
        return doc
    
    
    