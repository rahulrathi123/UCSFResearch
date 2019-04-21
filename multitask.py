from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path

import spacy
from spacy.util import minibatch, compounding

import pandas as pd
import csv


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir="model/", n_iter=5, n_texts=2000):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')

    # add label to text classifier
    textcat.add_label('s_imb')
    textcat.add_label('s_bal')
    textcat.add_label('s_miss')
    textcat.add_label('s_dir_pos')
    textcat.add_label('s_dir_neu')
    textcat.add_label('s_dir_neg')
    textcat.add_label('s_dir_miss')
    textcat.add_label('c_imb')
    textcat.add_label('c_bal')
    textcat.add_label('c_miss')
    textcat.add_label('c_dir_pos')
    textcat.add_label('c_dir_neu')
    textcat.add_label('c_dir_neg')
    textcat.add_label('c_dir_miss')

    (train_texts, train_cats), (dev_texts, dev_cats) = load_data()
    #print("LENGHTS")
    #print(len(train_texts))
    #print(len(train_cats))
    #print(len(dev_texts))
    #print(len(dev_cats))



    print("Using {} examples ({} training, {} evaluation)"
          .format(n_texts, len(train_texts), len(dev_texts)))
    train_data = list(zip(train_texts,
                          [{'cats': cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F', 'A', 'HA'))
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('{0:.5f}\t{1:.5f}\t{2:.5f}\t{3:.5f}'  # print a simple table
                  .format(scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f'], scores['textcat_a']))

    # test the trained model

    #print(test_text, doc.cats)


    #print(test_text2, doc.cats)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        trained_model = spacy.load(output_dir)

        with open('result.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Report Text', 'Predicted Label', 'Actual'])
            for i in range(len(dev_texts)):
                if (dev_texts[i] == ""):
                    continue
                impression = dev_texts[i]
                correct_label = dev_cats[i]
                doc2 = trained_model(dev_texts[i])
                writer.writerow([impression, doc2.cats, correct_label])

        '''for impression, cat in dev_texts, dev_cats:
            doc2 = trained_model(impression)
            print(test_text, doc2.cats)'''
        df = pd.read_csv("datawithlabels.csv", usecols = ["Report Text"])
        impressions = []
        for index, row in df.iterrows():
            impressions.append(row["Report Text"])

        #print(clean_impressions)
        cleaned_imps = clean_impressions(impressions)
        with open('datawithlabels.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sag_imb', 'sag_bal', 'sag_imb_miss', 'sag_dir_pos', 'sag_dir_neu', 'sag_dir_neg', 'sag_dir_miss', 'cor_imb', 'cor_bal', 'cor_imb_miss', 'cor_dir_pos', 'cor_dir_neu', 'cor_dir_neu', 'cor_dir_miss'])
            for i in range(len(cleaned_imps)):
                if (cleaned_imps[i] == ""):
                    continue
                doc2 = trained_model(cleaned_imps[i])
                print(doc2.cats)
                writer.writerow([doc2.cats["s_imb"], doc2.cats["s_bal"], doc2.cats["s_miss"], doc2.cats["s_dir_pos"], doc2.cats["s_dir_neu"], doc2.cats["s_dir_neg"], doc2.cats["s_dir_miss"], doc2.cats["c_imb"], doc2.cats["c_bal"], doc2.cats["c_miss"], doc2.cats["c_dir_pos"], doc2.cats["c_dir_neu"], doc2.cats["c_dir_neg"], doc2.cats["c_dir_miss"]])
                print(i)


def load_data(split=0.8):
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation
    '''train_data, _ = thinc.extra.datasets.imdb()
    print("this is the train_data")
    print(train_data)
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)

    cats = [{'POSITIVE': bool(y)} for y in labels]'''

    impressions = []
    imp_class = []
    s_dir = []
    c_dir = []
    df1 = pd.read_csv("scoliosis3class2.csv", usecols = ["Report Text", "Label", "S_Direction", "C_Direction"])

    docs1 = []
    i = 0

    for index, row in df1.iterrows():
        impressions.append(row["Report Text"])
        imp_class.append(row["Label"])
        s_dir.append(row["S_Direction"])
        c_dir.append(row["C_Direction"])
    #print(impressions)
    new_impressions = clean_impressions(impressions)
    #print(new_impressions)
    #print(impressions)
    #print(imp_class)
    train_data = list(zip(new_impressions, imp_class, s_dir, c_dir))
    #train_data = train_data[-0:]

    random.shuffle(train_data)
    texts, labels, s_dir, c_dir = zip(*train_data)

    n_iters = len(labels)

    cats = []
    for y in labels:
        if (y == 0):
            cats.append({"s_imb": False, "s_bal": False, "s_miss": True, "c_imb": False, "c_bal": False, "c_miss": True})
            continue
        if (y == 1):
            cats.append({"s_imb": False, "s_bal": True, "s_miss": False, "c_imb": False, "c_bal": False, "c_miss": True})
            continue
        if (y == 2):
            cats.append({"s_imb": True, "s_bal": False, "s_miss": False, "c_imb": False, "c_bal": False, "c_miss": True})
            continue
        if (y == 3):
            cats.append({"s_imb": False, "s_bal": False, "s_miss": True, "c_imb": False, "c_bal": True, "c_miss": False})
            continue
        if (y == 4):
            cats.append({"s_imb": False, "s_bal": False, "s_miss": True, "c_imb": True, "c_bal": False, "c_miss": False})
            continue
        if (y == 5):
            cats.append({"s_imb": False, "s_bal": True, "s_miss": False, "c_imb": False, "c_bal": True, "c_miss": False})
            continue
        if (y == 6):
            cats.append({"s_imb": True, "s_bal": False, "s_miss": False, "c_imb": False, "c_bal": True, "c_miss": False})
            continue
        if (y == 7):
            cats.append({"s_imb": False, "s_bal": True, "s_miss": False, "c_imb": True, "c_bal": False, "c_miss": False})
            continue
        if (y == 8):
            cats.append({"s_imb": True, "s_bal": False, "s_miss": False, "c_imb": True, "c_bal": False, "c_miss": False})
            continue

    i = -1
    for y in s_dir:
        i += 1
        if (y == 0):
            cats[i].update({"s_dir_pos": False, "s_dir_neu": False, "s_dir_neg": False, "s_dir_miss": True})
            continue
        if (y == 1):
            cats[i].update({"s_dir_pos": False, "s_dir_neu": False, "s_dir_neg": True, "s_dir_miss": False})            
            continue
        if (y == 2):
            cats[i].update({"s_dir_pos": False, "s_dir_neu": True, "s_dir_neg": False, "s_dir_miss": False})            
            continue
        if (y == 3):
            cats[i].update({"s_dir_pos": True, "s_dir_neu": False, "s_dir_neg": False, "s_dir_miss": False})            
            continue

    i = -1
    for y in c_dir:
        i += 1
        if (y == 0):
            cats[i].update({"c_dir_pos": False, "c_dir_neu": False, "c_dir_neg": False, "c_dir_miss": True})
            continue
        if (y == 1):
            cats[i].update({"c_dir_pos": False, "c_dir_neu": False, "c_dir_neg": True, "c_dir_miss": False})            
            continue
        if (y == 2):
            cats[i].update({"c_dir_pos": False, "c_dir_neu": True, "c_dir_neg": False, "c_dir_miss": False})            
            continue
        if (y == 3):
            cats[i].update({"c_dir_pos": True, "c_dir_neu": False, "c_dir_neg": False, "c_dir_miss": False})            
            continue

    print("PRINTING CATS")
    print(cats)

    df2 = pd.read_csv("QC_data.csv", usecols = ["Report_Text", "Classification", "S_Dir", "C_Dir"])
    test_impressions = []
    test_label = []
    test_S_Dir = []
    test_C_Dir = []
    for index, row in df2.iterrows():
        test_impressions.append(row["Report_Text"])
        test_label.append(row["Classification"])
        test_S_Dir.append(row["S_Dir"])
        test_C_Dir.append(row["C_Dir"])

    #POSSIBLE DEBUG BY PRINTING test_label
    
    test_cats = []
    for y in test_label:
        if (y == 0):
            test_cats.append({"s_imb": False, "s_bal": False, "s_miss": True, "c_imb": False, "c_bal": False, "c_miss": True})
            continue
        if (y == 1):
            test_cats.append({"s_imb": False, "s_bal": True, "s_miss": False, "c_imb": False, "c_bal": False, "c_miss": True})
            continue
        if (y == 2):
            test_cats.append({"s_imb": True, "s_bal": False, "s_miss": False, "c_imb": False, "c_bal": False, "c_miss": True})
            continue
        if (y == 3):
            test_cats.append({"s_imb": False, "s_bal": False, "s_miss": True, "c_imb": False, "c_bal": True, "c_miss": False})
            continue
        if (y == 4):
            test_cats.append({"s_imb": False, "s_bal": False, "s_miss": True, "c_imb": True, "c_bal": False, "c_miss": False})
            continue
        if (y == 5):
            test_cats.append({"s_imb": False, "s_bal": True, "s_miss": False, "c_imb": False, "c_bal": True, "c_miss": False})
            continue
        if (y == 6):
            test_cats.append({"s_imb": True, "s_bal": False, "s_miss": False, "c_imb": False, "c_bal": True, "c_miss": False})
            continue
        if (y == 7):
            test_cats.append({"s_imb": False, "s_bal": True, "s_miss": False, "c_imb": True, "c_bal": False, "c_miss": False})
            continue
        if (y == 8):
            test_cats.append({"s_imb": True, "s_bal": False, "s_miss": False, "c_imb": True, "c_bal": False, "c_miss": False})
            continue

    i = -1
    for y in test_S_Dir:
        i += 1
        if (y == 0):
            test_cats[i].update({"s_dir_pos": False, "s_dir_neu": False, "s_dir_neg": False, "s_dir_miss": True})
            continue
        if (y == 1):
            test_cats[i].update({"s_dir_pos": False, "s_dir_neu": False, "s_dir_neg": True, "s_dir_miss": False})            
            continue
        if (y == 2):
            test_cats[i].update({"s_dir_pos": False, "s_dir_neu": True, "s_dir_neg": False, "s_dir_miss": False})            
            continue
        if (y == 3):
            test_cats[i].update({"s_dir_pos": True, "s_dir_neu": False, "s_dir_neg": False, "s_dir_miss": False})            
            continue

    i = -1
    for y in test_C_Dir:
        i += 1
        if (y == 0):
            test_cats[i].update({"c_dir_pos": False, "c_dir_neu": False, "c_dir_neg": False, "c_dir_miss": True})
            continue
        if (y == 1):
            test_cats[i].update({"c_dir_pos": False, "c_dir_neu": False, "c_dir_neg": True, "c_dir_miss": False})            
            continue
        if (y == 2):
            test_cats[i].update({"c_dir_pos": False, "c_dir_neu": True, "c_dir_neg": False, "c_dir_miss": False})            
            continue
        if (y == 3):
            test_cats[i].update({"c_dir_pos": True, "c_dir_neu": False, "c_dir_neg": False, "c_dir_miss": False})            
            continue

    print("these should all match")
    print(len(test_impressions))
    print(len(cats))
    print(len(test_label))
    print(len(test_cats))

    print(test_cats[i])

    #print(len(texts))
    delete = []
    for i in range(len(texts)):
        if (texts[i].isspace() or texts[i] == ""):
            print("found")
            print(i)
            #print("blank")
            #print(texts[i])
            delete.append(i)
    i = len(delete) - 1
    while i >= 0:
        texts = texts[:delete[i]] + texts[delete[i]+1:]
        cats = cats[:delete[i]] + cats[delete[i]+1:]
        i -= 1

    #print("after deleting blanks")
    #print(len(texts))

    with open('cleaneddata.csv', mode='w') as data:
        data = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for i in range(len(texts)):
            data.writerow([texts[i]])
    #print("STORED CLEANED DATA")

    #split = int(len(texts) * split)

    

    #print("testing set length: ")
    #print(len(test_impressions))

    delete_indices = []

    flag = True

    duplicates = 0


    for i in range(len(test_impressions)):
        if (not flag):
            print("DID NOT FIND")
        print(i)    
        flag = False
        for j in range(len(texts)):
            if (compare(texts[j], test_impressions[i])):
                delete_indices.append(i)
                print(j)
                if (flag):
                    print("DUPLICATE FOUND")
                    duplicates += 1
                flag = True

    print("Number of duplicates: ")
    print(duplicates)

    print("length of QC ")
    print(len(test_impressions))

    print("these many found to delete: ") 
    print(len(delete_indices))

    for i in range(len(delete_indices)):
        texts = texts[:delete_indices[i]] + texts[delete_indices[i]+1:]
        cats = cats[:delete_indices[i]] + cats[delete_indices[i]+1:]        


    '''
    deletestuff = []
    for i in range(len(texts)):
        for j in range(len(test_impressions)):
            if (compare(texts[i], test_impressions[j])):
                deletestuff.append(i)
                print(j)
                if (flag):
                    print("DUPLICATE FOUND AFTER DELETION")
                    duplicates += 1
                flag = True
    print("Checked for duplicates after deletion")
    DO THIS AFTER TRAINING

    '''
    with open('traintestsplit.csv', mode='w') as data:
        data = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data.writerow("Training Set")
        for i in range(len(texts)):
            data.writerow([texts[i]])
        
        data.writerow("Testing Set")
        
        for i in range(len(test_impressions)):
            data.writerow([test_impressions[i]])

    print("IMPORTANT LENGTHS")
    print(len(texts))
    print(len(cats))
    print(len(test_impressions))
    print(len(test_cats))

    return (texts, cats), (test_impressions, test_cats)

def clean_impressions(impressions):
    delete_words = ["IMPRESSION:\n", "END", "OF", "IMPRESSION:\n", "\n"]
    end_of_impression = ["Report", "dictated"]

    impressions_list = []
    for impression in impressions:
        sentence = []
        word = []
        for character in impression:
            if (character != " "):
                word.append(character)
            else:
                sentence.append(word)
                word = []

        sentence2 = []
        word2 = []

        for word in sentence:
            word2 = ''.join(word)
            if (word2 == "Report"):
                break
            if (word2 == "//Impression"):
                break
            if (word2 == "//Impressio"):
                break
            sentence2.append(word2)

        sentence3 = []
        for word in sentence2:
            if word not in delete_words:
                sentence3.append(word)
        sentence4 = ' '.join(sentence3)
        impressions_list.append(sentence4)
    return impressions_list

def compare(trainImp, testImp):

    train = []
    test = []
    for character in trainImp:
        train.append(character)
    for character in testImp:
        test.append(character)

    i = 0
    j = 0
    while ((i < len(train)) and (j < len(test))):
        #print("train: ", train[i])
        #print("test: ", test[j])
        if (train[i] == test[j]):
            i += 1
            j += 1
            continue
        if (train[i] == " "):
            i += 1
            continue
        if (test[j] == " "):
            j += 1
            continue
        return False

    return True

def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        #print(i)
        #print(len(cats))
        #print(cats)
        #print(i)
        #print(cats[i])

        gold = cats[i]


        for label, score in doc.cats.items():

            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.

            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.

            elif score < 0.5 and gold[label] < 0.5:
                tn += 1

            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1


    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score, 'textcat_a': accuracy}


if __name__ == '__main__':
    plac.call(main)