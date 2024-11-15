
import pandas as pd
import networkx as nx
from networks import make_net, make_net_from_df, save_net, get_net, extract_forum_data, show_net, visualize_network, filter_length, filter_users
from earlyadopters import get_early_adopters
from feature_extraction import get_features, test, test2
from tests import get_avg_time_difference, average_time_everything
from balance import prepare
from datetime import datetime, timedelta
from IPython.display import display
import datetime as dt


mult = [4]
alpha = 10
minlength = 10
minPosts = 5
minThreads = 2
for m in mult:
    #4, 8
    forums = [4]
    for forum in forums:
        print(f'Doing Forum {forum}')
        
        df = extract_forum_data(forum, minlength)
        df = filter_length(df, minlength)
        df = filter_users(df, minPosts, minThreads)
        df = df.sort_values(by=['dateadded_post'])
        #display(df)

        get_avg_time_difference(df)
        average_time_everything(df)


        #9186
        #9288      40
        #11157     40
        #print(df['topic_id'].value_counts().to_string())

        #df = df.query('topic_id == 9288 or topic_id == 11157')
        #LIMIT YOUR DATA / QUERY BY ALPHA !!!!
        #pkp = f'pickleX{forum}.p'        
        #net = make_net_from_df(df, 7)
        #pkp = save_net(net, forum)
        
        #visualize_network(net, forum)

        #show_net(pkp, forum)        

        # run query to retrieve topics id's where above > size and > size/2
        csc, ncsc, tcsc, tncsc = get_early_adopters(df, forum, alpha, alpha*m) # set threshold here    
        
        # Prepare sets
        split = 0.8 # not over 1
        train_threads, train_times, test_threads, test_times = prepare(csc, ncsc, tcsc, tncsc, split, 2) # 1 = both balanced; 2 = train balanced, test imbalanced; 3 = both imbalanced
        #train_threads, train_times, test_threads, test_times = csc, ncsc, tcsc, tncsc 

        # Get features -> save as pdf
        #get_avg_time_difference(train_threads, train_times, df)
        train_df = get_features(train_threads, train_times, df, sigma=14)
        train_df.to_csv(f'Forum{forum}data_{m}x_train.csv', header = True, index = False)
        del train_df

        # Get features -> save as pdf
        test_df = get_features(test_threads, test_times, df, sigma=14)
        test_df.to_csv(f'Forum{forum}data_{m}x_test.csv', header = True, index = False)
        del test_df

#7, 14, 60, 140, 190, 200
#14, 60
#Go with 14, try classifier
#Adaboost, decision tree, randomforest
#include page rank, no harm
#for now lets use only the max (non penalized, 2nd) value of diameter and avg shortest length
#make note of the class distribution
    #argument called stratify