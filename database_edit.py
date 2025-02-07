from networks import get_db_connection_to_upload, upload_to_database
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.utils import resample

from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display

def connection():
    engine = get_db_connection_to_upload()
    query = "SELECT * FROM all_features_checkpoint_4"
    with engine.connect() as conn:
        df = pd.read_sql(
            sql=query,
            con=conn.connection,
        )
    engine.close()
    return df
def classify(model, x_train_set, y_train_set, x_test_set, y_test_set):
    model.fit(x_train_set, y_train_set)
    y_predict = model.predict(x_test_set)
    # accuracy = accuracy_score(y_test_set, y_predict)
    ps = precision_score(y_test_set, y_predict)
    rs = recall_score(y_test_set, y_predict)
    fs = f1_score(y_test_set, y_predict)

    return ps, rs, fs

def make_training_set(xs, ys, train_index, test_index):
    x_train = xs.iloc[train_index]
    y_train = ys.iloc[train_index]

    x_test = xs.iloc[test_index]
    y_test = ys.iloc[test_index]

    #baseline
    x_train_final = x_train
    y_train_final = y_train

    #smote implementation
    #smote = SMOTE(random_state=42, k_neighbors=5)
    #x_train_final, y_train_final = smote.fit_resample(x_train, y_train)

    #smote enn implementation
    #smote_enn = SMOTEENN(random_state=42)
    #x_train_final, y_train_final = smote_enn.fit_resample(x_train, y_train)

    #downsampling implemnetation
    x_train_final, y_train_final = downsample(x_train, y_train)

    #filtering implementation

    return x_train_final, y_train_final, x_test, y_test

def variables():
    # early adopters only:
    e_only = False
    one_only = False

    title = "scores_twohop_downsample"

    split_count = 5

    return e_only, one_only, title, split_count

def downsample(x_train, y_train):
    # Combine the features and labels into a single dataset
    train_data = pd.concat([x_train, y_train], axis=1)

    # Separate the majority and minority classes
    majority_class = train_data[train_data[y_train.name] == 0]
    minority_class = train_data[train_data[y_train.name] == 1]

    # Downsample the majority class
    majority_downsampled = resample(majority_class,
                                    replace=False,  # without replacement
                                    n_samples=len(minority_class),  # match minority class size
                                    random_state=42)

    # Combine the downsampled majority class with the minority class
    downsampled_data = pd.concat([majority_downsampled, minority_class])

    # Separate the features and labels again
    x_train_resampled = downsampled_data.drop(columns=[y_train.name])
    y_train_resampled = downsampled_data[y_train.name]

    return x_train_resampled, y_train_resampled

def main():

    df = connection()
    forums = df['forum_id'].unique()
    topics = df['topic_id'].unique()
    alphas = df['alpha'].unique()
    betas = df['beta'].unique()
    content_lengths = df['min_post_content_length'].unique()
    post_counts = df['min_user_post_count'].unique()
    thread_counts = df['min_user_thread_count'].unique()
    # sigmas = df['Sigma'].unique()
    sigmas = [[30, 14], [14, 7]]
    deltaT = [60, 90, 120, 150, 180]
    #deltaT = [999999]

    #variables to change
    e_only, one_only, name, split_count = variables()


    data = []

    for f in forums:
        for a in alphas:
            for b in betas:
                for c in content_lengths:
                    for p in post_counts:
                        for t in thread_counts:
                            for s in sigmas:

                                conditions = (df['forum_id'] == f) & (df['alpha'] == a) & (df['beta'] == b) & (df['min_post_content_length'] == c) & (df['min_user_post_count'] == p) & (df['min_user_thread_count'] == t) & (df['sigma_sus'] == s[0]) & (df['sigma_fos'] == s[1])
                                dataframe = df[conditions]

                                if dataframe.empty:
                                    #print(f, a, b, c, p, t, s)
                                    #print(f"Empty for {f, a, b, c, p, t, s}")
                                    continue

                                for dt in deltaT:
                                    print(dt)
                                    deltaT_dataset = dataframe.iloc[:, 9:]
                                    dataset = deltaT_dataset.iloc[:, 1:]
                                    #display(dataset)
                                    for index, row in deltaT_dataset.iterrows():
                                        if row['delta_t'] > dt or row['delta_t'] == -1:
                                            #print(row['delta_t'])
                                            #print(dt)
                                            dataset.at[index, 'class_label'] = 0

                                    if e_only:
                                        xs = dataset.iloc[:, :36] #36 for early adopters only
                                    elif one_only:
                                        xs = dataset.iloc[:, :52]
                                    else:
                                        xs = dataset.iloc[:, :-1]
                                    ys = dataset.iloc[:, -1]

                                    counts = ys.value_counts()
                                    if 0 not in counts.index or 1 not in counts.index:
                                        print(f"Error for {dt}: Value 0 or 1 DNE for {f, a, b, c, p, t, s}")
                                        continue
                                    else:
                                        print(f"Running for {f, a, b, c, p, t, s}")
                                    ys_negative_count = counts[0]
                                    ys_positive_count = counts[1]
                                    if ys_positive_count == 1 or ys_negative_count == 1:
                                        print(f"Error for {dt}: Only 1 pos/neg case")
                                        continue
                                    print(ys_negative_count, ys_positive_count)

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

                                    try:
                                        for i, (train_index, test_index) in enumerate(test):
                                            x_train, y_train, x_test, y_test = make_training_set(xs, ys, train_index, test_index)
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



                                    except ValueError as e:
                                        # Skip the current delta_t loop if SMOTE fails for any fold
                                        print(f"Error for {dt}: SMOTE failed: {e}")
                                        continue

                                    print(f"Success for {dt}")
                                    data.append([f, a, b, c, p, t, s, dt, 'RandomForest', rfc_p/split_count, rfc_r/split_count, rfc_f/split_count, ys_positive_count, ys_negative_count])
                                    data.append([f, a, b, c, p, t, s, dt, 'DecisionTree', dfc_p/split_count, dfc_r/split_count, dfc_f/split_count, ys_positive_count, ys_negative_count])
                                    data.append([f, a, b, c, p, t, s, dt, 'AdaBoost', abc_p/split_count, abc_r/split_count, abc_f/split_count, ys_positive_count, ys_negative_count])

    pdf = pd.DataFrame(data, columns=['forum_id', 'alpha', 'beta', 'min_post_content_length',
                                          'min_user_post_count', 'min_user_thread_count', 'sigmas', 'delta_t', 'classiifer',
                                          'precision', 'recall', 'f1_score', 'positive', 'negative'])
    upload_to_database(pdf, name)

if __name__ == "__main__":
    main()
