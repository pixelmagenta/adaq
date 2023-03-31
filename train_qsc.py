from parc3corpus import Parc3Corpus
from sklearn_crfsuite import metrics
from spacy.training import tags_to_entities, iob_to_biluo, biluo_to_iob, offsets_to_biluo_tags
import verb_cue_classifier
import content_classifier
import source_classifier
import sklearn_crfsuite
import spacy
import tqdm
import pickle
import argparse

def create_training_set(examples, nlp, target_label):
    X = []
    y = []
    for example in tqdm.tqdm(examples):
        doc = example.reference
        nlp(doc)
        tags = offsets_to_biluo_tags(doc, [(span.start_char, span.end_char, 'source')
                                        for span in doc._.source_spans])
        tags = biluo_to_iob(tags)
        for sent in doc.sents:
            X.append([t._.source_features for t in sent])
            y.append([tags[t.i] for t in sent])
    return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('-l', '--lang',
                    action='store_true')

    args = parser.parse_args()
    
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('source_classifier_features')
    if args.lang:
        print('enabling lang features')
        nlp.add_pipe('source_classifier_lang_features')

    print('parsing training set')
    c = Parc3Corpus(f'./data/parc3/train/')
    X_train, y_train = create_training_set(c(nlp), nlp, 'source')
    print('parsing test set')
    c = Parc3Corpus(f'./data/parc3/test/')
    X_test, y_test = create_training_set(c(nlp), nlp, 'source')
    print('parsing dev set')
    c = Parc3Corpus(f'./data/parc3/dev/')
    X_dev, y_dev = create_training_set(c(nlp), nlp, 'source')
    with open('qsc_train.dmp', 'wb') as f:
        pickle.dump([X_train, y_train, X_test, y_test, X_dev, y_dev], f)

    def iob_to_entities(arr):
        arr = [x + '-X' if x in ['B', 'I'] else x for x in arr]
        return tags_to_entities(iob_to_biluo(arr))

    def f1_bbc(y_true, y_pred):
        tp = 0
        fp = 0
        true_count = 0
        for doc_pred, doc_true in zip(y_pred, y_true):
            pred_ents = iob_to_entities(doc_pred)
            true_ents = iob_to_entities(doc_true)
            true_count += len(true_ents)
            for pred in pred_ents:
                if pred in true_ents:
                    tp += 1
                else:
                    fp += 1
        precision = tp / (tp + fp)
        recall = tp / true_count
        return 2 * precision * recall / (precision + recall), precision, recall

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=1000,
        all_possible_transitions=True,
        verbose=True
    )
    crf.fit(X_train, y_train, X_dev, y_dev)

    labels = list(crf.classes_)
    labels.remove('O')

    y_pred = crf.predict(X_test)
    print('F1 on test:', metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))
    print('BBC F1 on test:', f1_bbc(y_test, y_pred))
    y_pred = crf.predict(X_dev)
    print('BBC F1 on dev:', f1_bbc(y_dev, y_pred))

    with open(args.out, 'wb') as f:
        pickle.dump(crf, f)