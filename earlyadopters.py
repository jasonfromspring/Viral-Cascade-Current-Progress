
from connect import get
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pickle as pk
import random as rd
from IPython.display import display



def get_early_adopters(df, forum, alpha, beta, roots):
    casc = {}
    noncasc = {}
    timecsc = {}
    timencsc = {}
    timebcsc = {}

    gdf = df[["user_id", "topic_id", "post_id", "dateadded_post"]].sort_values(by=['dateadded_post'])
    '''
    #Final Version:
    
    #gdf = get(tbl, 'users_id, topics_id, posts_id, posted_date', None, None) # entire forum
    # get topics and distinct user count
    gdf = get('posts as "p" inner join topics as "t" on t.topic_id = p.topic_id', 
              'p.user_id, p.topic_id, p.post_id, p.dateadded_post', 
              f"forum_id = {forum} and classification2_topic >= 0.5 and length(content_post) > 10 group by p.user_id, p.post_id, p.topic_id order by p.dateadded_post asc", None)
                #"group by p.user_id, p.post_id, p.topic_id order by p.dateadded_post asc"
    #query = f"select p.topic_id, count(distinct p.user_id) " \
    #       f"from posts p inner join topics t on t.topic_id = p.topic_id " \
    #        f"where forum_id = {forum_id} and classification2_topic >= 0.5 and p.dateadded_post > '{start_date_text}' and length(content_post) > 10 group by p.topic_id"
    display(gdf)
    #tdf = get('t_posts', 'topics_id as topics, count (distinct users_id) as cnt', None, f'where forums_id = {forum} group by topics_id')
    tdf = get('posts as p inner join topics as t on t.topic_id = p.topic_id', 
              'p.topic_id as topics, count (distinct p.user_id) as cnt', None, 
              f'where forum_id = {forum} and classification2_topic >= 0.5 and length(content_post) > 10 group by p.topic_id')
                #group by p.topic_id
    display(tdf)
    '''
    tdf = df[['topic_id', 'user_id']].groupby(['topic_id'], as_index=False).nunique().rename(columns={'topic_id':'topics', 'user_id':'cnt'})
    #display(tdf)

    '''
    #Test Dataset Version
    gdf = get('posts as "p" inner join topics as "t" on t.topic_id = p.topic_id', 
              'p.user_id, p.topic_id, p.post_id, p.dateadded_post', 
              f"forum_id = {forum} and classification2_topic >= 0.5 and length(content_post) > 10 and (t.topic_id = 7193 or t.topic_id = 7224 or t.topic_id = 7226 or t.topic_id = 7229 or t.topic_id = 7256 or t.topic_id = 7278 or t.topic_id = 7298 or t.topic_id = 7318 or t.topic_id = 7377 or t.topic_id = 7469 or t.topic_id = 7182 or t.topic_id = 7192 or t.topic_id = 7194 or t.topic_id = 7195 or t.topic_id = 7221 or t.topic_id = 7222 or t.topic_id = 7225) group by p.user_id, p.post_id, p.topic_id order by p.dateadded_post asc", None)
    
    tdf = get('posts as p inner join topics as t on t.topic_id = p.topic_id', 
              'p.topic_id as topics, count (distinct p.user_id) as cnt', None, 
              f'where forum_id = {forum} and classification2_topic >= 0.5 and length(content_post) > 10 and (t.topic_id = 7193 or t.topic_id = 7224 or t.topic_id = 7226 or t.topic_id = 7229 or t.topic_id = 7256 or t.topic_id = 7278 or t.topic_id = 7298 or t.topic_id = 7318 or t.topic_id = 7377 or t.topic_id = 7469 or t.topic_id = 7182 or t.topic_id = 7192 or t.topic_id = 7194 or t.topic_id = 7195 or t.topic_id = 7221 or t.topic_id = 7222 or t.topic_id = 7225) group by p.topic_id')
    
    '''
    # get all topics that reach n
    qdf = tdf[(tdf['cnt'] >= beta)]['topics'].tolist()
    # get all topics that reach n/2 but not n
    qdf2 = tdf[(tdf['cnt'] >= alpha) & (tdf['cnt'] < beta)]['topics'].tolist() # need to filter further
    
    del tdf
    # get positive user lists for each category
    # get times as well
    
    for t in qdf:
        usrs = gdf[gdf['topic_id'] == t]['user_id'].tolist() # all pos cases
        true_root = roots[t]
        if(true_root == usrs[0]):
            tms = gdf[gdf['topic_id'] == t]['dateadded_post'].tolist() # all pos times, need to verify
            betaTime = tms[beta-1]
            index = 0
            casc_users = {}
            lst = []
            lst2 = []
            while len(casc_users) < alpha:
                if usrs[index] not in casc_users.keys(): # if usr != i in list([0])
                    casc_users[usrs[index]] = 1 #tms[index] # first alpha users, with their first time
                    lst.append(usrs[index])
                    lst2.append(tms[index])
                index += 1

            casc[t] = lst
            timecsc[t] = lst2
            timebcsc[t] = betaTime
        #else:
        #    print(t)
        #print(f'total in casc is {len(set(usrs))}')
        

    for t2 in qdf2:
        usrs2 = gdf[gdf['topic_id'] == t2]['user_id'].tolist()      #[:alpha]
        true_root = roots[t2]
        if(true_root == usrs2[0]):
            tms2 = gdf[gdf['topic_id'] == t2]['dateadded_post'].tolist()  #[:alpha]
            noncasc_users = {}
            lst3 = []
            lst4 = []
            for z in range(len(usrs2)):
                if len(noncasc_users) < alpha:
                    if usrs2[z] not in noncasc_users.keys():
                        noncasc_users[usrs2[z]] = tms2[z] # first alpha users, with their first time
                        lst3.append(usrs2[z])
                        lst4.append(tms2[z])
            
            noncasc[t2] = lst3
            timencsc[t2] = lst4
            #print(f'total in ncasc is {len(set(usrs2))}')
            #avg.append(len(set(gdf[gdf['topics_id'] == t2]['users_id'].tolist())))
            #nonsize[t2] = len(set(gdf[gdf['topics_id'] == t2]['users_id'].tolist()))
            #ns[t2] = list(set(gdf[gdf['topics_id'] == t2]['users_id'].tolist()))
        #else:
        #    print(t2)
    
    del gdf

    return casc, noncasc, timecsc, timencsc, timebcsc