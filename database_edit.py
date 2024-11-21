from networks import get_db_connection_to_upload, upload_to_database
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display


engine = get_db_connection_to_upload()
query = "SELECT * FROM features"
with engine.connect() as conn:
    df = pd.read_sql(
        sql=query,
        con=conn.connection,
    )
engine.close()

forums = df['Forum'].unique()
topics = df['Topic'].unique()
alphas = df['Alpha'].unique()
betas = df['Beta'].unique()
content_lengths = df['Min Post Content Length'].unique()
post_counts = df['Min User Post Count'].unique()
thread_counts = df['Min User Thread Count'].unique()
sigmas = df['Sigma'].unique()

print(forums, topics, alphas, betas, content_lengths, post_counts, thread_counts, sigmas)

data = []

def classify(model, x_train_set, y_train_set, x_test_set, y_test_set):
    model.fit(x_train_set, y_train_set)
    y_predict = model.predict(x_test_set)
    # accuracy = accuracy_score(y_test_set, y_predict)
    ps = precision_score(y_test_set, y_predict)
    rs = recall_score(y_test_set, y_predict)
    fs = f1_score(y_test_set, y_predict)
    return ps, rs, fs

for f in forums:
    for a in alphas:
        for b in betas:
            for c in content_lengths:
                for p in post_counts:
                    for t in thread_counts:
                        for s in sigmas:

                            conditions = (df['Forum'] == f) & (df['Alpha'] == a) & (df['Beta'] == b) & (df['Min Post Content Length'] == c) & (df['Min User Post Count'] == p) & (df['Min User Thread Count'] == t) & (df['Sigma'] == s)
                            print(f, a, b, c, p, t, s)
                            dataset = df[conditions]
                            if dataset.empty:
                                break
                            dataset = dataset.iloc[:, 8:]
                            xs = dataset.iloc[:, :-1]
                            ys = dataset.iloc[:, -1]

                            ys_positive_count = ys.value_counts()[1]
                            ys_negative_count = ys.value_counts()[0]

                            split_count = 5
                            skf = StratifiedKFold(n_splits=split_count)
                            test = skf.split(xs, ys)

                            rfc_p = 0
                            rfc_r = 0
                            rfc_f = 0
                            dfc_p = 0
                            dfc_r = 0
                            dfc_f = 0
                            abc_p = 0
                            abc_r = 0
                            abc_f = 0

                            for i, (train_index, test_index) in enumerate(test):
                                
                                x_train = xs.iloc[train_index]
                                y_train = ys.iloc[train_index]

                                x_test = xs.iloc[test_index]
                                y_test = ys.iloc[test_index]
                                
                                # x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=42, stratify=ys)
    
                                rfc = RandomForestClassifier(random_state=42)
                                precision, recall, f1 = classify(rfc, x_train, y_train, x_test, y_test)
                                rfc_p += precision
                                rfc_r += recall
                                rfc_f += f1

                                dfc = DecisionTreeClassifier(random_state=42)
                                precision, recall, f1 = classify(dfc, x_train, y_train, x_test, y_test)
                                dfc_p += precision
                                dfc_r += recall
                                dfc_f += f1

                                abc = AdaBoostClassifier(algorithm='SAMME', random_state=42)
                                precision, recall, f1 = classify(abc, x_train, y_train, x_test, y_test)
                                abc_p += precision
                                abc_r += recall
                                abc_f += f1

                            data.append([f, a, b, c, p, t, s, 'RandomForest', rfc_p/split_count, rfc_r/split_count, rfc_f/split_count, ys_positive_count, ys_negative_count])
                            data.append([f, a, b, c, p, t, s, 'DecisionTree', dfc_p/split_count, dfc_r/split_count, dfc_f/split_count, ys_positive_count, ys_negative_count])
                            data.append([f, a, b, c, p, t, s, 'AdaBoost', abc_p/split_count, abc_r/split_count, abc_f/split_count, ys_positive_count, ys_negative_count])

pdf = pd.DataFrame(data, columns=['Forum', 'Alpha','Beta','Min Content Length', 'Min User Posts', 'Min User Threads', 'Sigma', 'Classifier',
                                  "Precision", 'Recall', 'F1 Score', 'Positive', 'Negative'])
upload_to_database(pdf, 'scores_test')

