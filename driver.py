
import pandas as pd
import networkx as nx
from networks import make_net, make_net_from_df, save_net, get_net, extract_forum_data, show_net, visualize_network, filter_length, filter_users, upload_to_database
from earlyadopters import get_early_adopters
from feature_extraction import get_features, test, test2
from tests import get_avg_time_difference, average_time_everything
from balance import prepare
from datetime import datetime, timedelta
from IPython.display import display
import datetime as dt


alpha_values = [10, 20, 30, 40, 50]
beta_multipliers = [2, 3, 4, 5, 10]
content_length_thresholds = [0, 2, 10]
min_posts_values = [1, 5, 10]
min_threads_values = [1, 2, 5]
sigmas = [7, 14, 30]
forums = [4,8,2,9,11]


#temporary:
sigmas = [30]
beta_multipliers = [3, 4, 5, 10]
"""
alpha_values = [10]
beta_multipliers = [4]
min_posts_values = [10,1]
min_posts_values = [5,10]
min_threads_values = [2,3]
forums = [4,8]
sigmas = [7]
"""

for forum in forums:
    print(f'Doing Forum {forum}')

    for minLength in content_length_thresholds:
        print(f"Min content length filter = {minLength}")

        for minPosts in min_posts_values:
            print(f'User post minimum: {minPosts}')

            for minThreads in min_threads_values:
                print(f'User thread minimum: {minThreads}')
                
                df = extract_forum_data(forum, minLength)
                df = filter_length(df, minLength)
                df = filter_users(df, minPosts, minThreads)
                df = df.sort_values(by=['dateadded_post'])

                for alpha in alpha_values:
                    print(f'\nDoing alpha {alpha}')

                    for m in beta_multipliers:
                        print(f'Beta multiplier {m}, beta = {alpha*m}')

                        csc, ncsc, tcsc, tncsc = get_early_adopters(df, forum, alpha, alpha*m)

                        for sigma in sigmas:
                            print(f"Sigma = {sigma}\n")
                            filters = [forum, alpha, alpha*m, minLength, minPosts, minThreads]
                            results = get_features(csc, ncsc, tcsc, tncsc, df, sigma, filters)
                            #results.to_csv(f'Forum{forum}data_{m}x.csv', header = True, index = False)
                            upload_to_database(results, 'features')

                        sigmas = [7, 14, 30]

                    beta_multipliers = [2, 3, 4, 5, 10]
                    
'''
for alpha in alpha_values:
    print(f'\nDoing alpha {alpha}')
    for m in beta_multipliers:
        print(f'Beta multiplier {m}, beta = {alpha*m}')
        for minLength in min_posts_values:
            print(f"Min content length filter = {minLength}")
            for minPosts in min_posts_values:
                print(f'User post minimum: {minPosts}')
                for minThreads in min_threads_values:
                    print(f'User thread minimum: {minThreads}')
                    for sigma in sigmas:
                        print(f"Sigma = {sigma}\n")
                        for forum in forums:
                            print(f'Doing Forum {forum}')

                            
                            df = extract_forum_data(forum, minLength)
                            df = filter_length(df, minLength)
                            df = filter_users(df, minPosts, minThreads)
                            df = df.sort_values(by=['dateadded_post'])
                            #display(df)

                            #get_avg_time_difference(df)
                            #average_time_everything(df)


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
                            #split = 0.8 # not over 1
                            #train_threads, train_times, test_threads, test_times = prepare(csc, ncsc, tcsc, tncsc, split, 2) # 1 = both balanced; 2 = train balanced, test imbalanced; 3 = both imbalanced
                            #train_threads, train_times, test_threads, test_times = csc, ncsc, tcsc, tncsc 

                            # Get features -> save as pdf
                            #get_avg_time_difference(train_threads, train_times, df)
                            filters = [forum, alpha, alpha*m, minLength, minPosts, minThreads]
                            results = get_features(csc, ncsc, tcsc, tncsc, df, sigma, filters)
                            #results.to_csv(f'Forum{forum}data_{m}x.csv', header = True, index = False)
                            upload_to_database(results)
                            #del results


                    #7, 14, 60, 140, 190, 200
                    #14, 60
                    #Go with 14, try classifier
                    #Adaboost, decision tree, randomforest
                    #include page rank, no harm
                    #for now lets use only the max (non penalized, 2nd) value of diameter and avg shortest length
                    #make note of the class distribution
                        #argument called stratify
'''