import spacy
import pandas as pd
import csv

def main(output_dir = "model"):
    print("Loading from", output_dir)
    nlp = spacy.load(output_dir)

    if 'textcat' not in nlp.pipe_names:
        print("IF SATISFIED")
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    else:
        print("ELSE")
        textcat = nlp.get_pipe('textcat')

#last argument should be different each time we call the evaluate function
    df = pd.read_csv("QC_data.csv", usecols = ["Report_Text", "s_imb_class", "c_imb_class", "S_Dir", "C_Dir"])
    impressions = []
    s_imb_values = []
    c_imb_values = []
    s_dir_values = []
    c_dir_values = []
    test_s_imb = []
    test_c_imb = []
    test_s_dir = []
    test_c_dir = []
    i = 0
    for index, row in df.iterrows():
        print(i)
        impressions.append(row["Report_Text"])
        s_imb_values.append(row["s_imb_class"])
        c_imb_values.append(row["c_imb_class"])
        s_dir_values.append(row["S_Dir"])
        c_dir_values.append(row["C_Dir"])
        i += 1
    i = 0
    for imp in s_imb_values:
        print(i)
        i += 1
        if (imp == 0):
            test_s_imb.append({"s_imb": False, "s_bal": False, "s_miss": True})
            continue
        if (imp == 1):
            test_s_imb.append({"s_imb": False, "s_bal": True, "s_miss": False})
            continue
        if (imp == 2):
            test_s_imb.append({"s_imb": True, "s_bal": False, "s_miss": False})
            continue
    i = 0
    for imp in c_imb_values:
        print(i)
        i += 1
        if (imp == 0):
            test_c_imb.append({"c_imb": False, "c_bal": False, "c_miss": True})
            continue
        if (imp == 1):
            test_c_imb.append({"c_imb": False, "c_bal": True, "c_miss": False})
            continue
        if (imp == 2):
            test_c_imb.append({"c_imb": True, "c_bal": False, "c_miss": False})
            continue
    i = 0
    for imp in s_dir_values:
        print(i)
        i += 1
        if (imp == 0):
            test_s_dir.append({"s_dir_pos": False, "s_dir_neu": False, "s_dir_neg": False, "s_dir_miss": True})
            continue
        if (imp == 1):
            test_s_dir.append({"s_dir_pos": False, "s_dir_neu": False, "s_dir_neg": True, "s_dir_miss": False})
            continue
        if (imp == 2):
            test_s_dir.append({"s_dir_pos": False, "s_dir_neu": True, "s_dir_neg": False, "s_dir_miss": False})
            continue
        if (imp == 3):
            test_s_dir.append({"s_dir_pos": True, "s_dir_neu": False, "s_dir_neg": False, "s_dir_miss": False})
            continue
    i = 0
    for imp in c_dir_values:
        print(i)
        i += 1
        if (imp == 0):
            test_c_dir.append({"c_dir_pos": False, "c_dir_neu": False, "c_dir_neg": False, "c_dir_miss": True})
            continue
        if (imp == 1):
            test_c_dir.append({"c_dir_pos": False, "c_dir_neu": False, "c_dir_neg": True, "c_dir_miss": False})
            continue
        if (imp == 2):
            test_c_dir.append({"c_dir_pos": False, "c_dir_neu": True, "c_dir_neg": False, "c_dir_miss": False})
            continue
        if (imp == 3):
            test_c_dir.append({"c_dir_pos": True, "c_dir_neu": False, "c_dir_neg": False, "c_dir_miss": False})
            continue


    print("S_IMB")
    scores1 = evaluate(nlp.tokenizer, textcat, impressions, test_s_imb)
    print('{0:.5f}\t{1:.5f}\t{2:.5f}\t{3:.5f}'  # print a simple table
        .format(scores1['textcat_p'],
            scores1['textcat_r'], scores1['textcat_f'], scores1['textcat_a']))
    print("C_IMB")
    scores2 = evaluate(nlp.tokenizer, textcat, impressions, test_c_imb)
    print('{0:.5f}\t{1:.5f}\t{2:.5f}\t{3:.5f}'  # print a simple table
        .format(scores2['textcat_p'],
            scores2['textcat_r'], scores2['textcat_f'], scores2['textcat_a']))
    print("S_DIR")

    scores3 = evaluate(nlp.tokenizer, textcat, impressions, test_s_dir)
    print('{0:.5f}\t{1:.5f}\t{2:.5f}\t{3:.5f}'  # print a simple table
        .format(scores3['textcat_p'],
            scores3['textcat_r'], scores3['textcat_f'], scores3['textcat_a']))
    print("C_DIR")

    scores4 = evaluate(nlp.tokenizer, textcat, impressions, test_c_dir)
    print('{0:.5f}\t{1:.5f}\t{2:.5f}\t{3:.5f}'  # print a simple table
        .format(scores4['textcat_p'],
            scores4['textcat_r'], scores4['textcat_f'], scores4['textcat_a']))


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
        if (i == 0):
            print(gold)
        modified_result = {}

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

main()