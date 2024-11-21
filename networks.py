from connect import get
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import math
from IPython.display import display
from urllib.parse import quote_plus


def get_db_connection():
    password = quote_plus('dw.@2')
    engine = create_engine(f'postgresql://postgres:{password}@localhost:5432/darkweb_markets_forums')
    return engine.connect()

def get_db_connection_to_upload():
    password = quote_plus('dw.@2')
    engine = create_engine(f'postgresql://postgres:{password}@localhost:5432/viral_cascade')
    return engine.connect()

def upload_to_database(df, table_name):
    conn = get_db_connection_to_upload()
    try:
        df.to_sql(table_name, conn, if_exists='append', index=False)
        print(f"Data appended successfully to the table '{table_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    conn.close()

def extract_forum_data(forum_id, min_post_length):
    engine = get_db_connection()
    query = """
    SELECT posts.user_id, posts.topic_id, posts.post_id, posts.dateadded_post, LENGTH(posts.content_post) AS post_length, topics.dateadded_topic AS thread_start_date, 
    topics.classification2_topic
    FROM posts
    INNER JOIN topics ON posts.topic_id = topics.topic_id
    WHERE topics.forum_id = %s AND topics.classification2_topic >= %s
    """

    #df = pd.read_sql(query, conn, params=(forum_id, 0.5))
    with engine.connect() as conn:
        df = pd.read_sql(
            sql=query,
            con=conn.connection,
            params=(forum_id, 0.5)
    )

    engine.close()
    return df 

def filter_length(df, min_post_length):
    
    df['dateadded_post'] = pd.to_datetime(df['dateadded_post'], utc=True)
    df['thread_start_date'] = pd.to_datetime(df['thread_start_date'], utc=True)
    
    # Filter posts by minimum length
    df = df[df['post_length'] > min_post_length]
    
    return df

def filter_users(df, min_posts, min_threads):
    
    
    """
    filter by users that have:
    posted at least 5 posts
    posted at least in 2 distinct threads
    all these posts are at least 2 characters
    """
    '''
    # Filter posts by minimum length
    count_freq = dict(df['user_id'].value_counts())
    df['count_freq'] = df['user_id']
    df['count_freq'] = df['count_freq'].map(count_freq)
    df = df[df.count_freq>=min_posts]
    '''
    df = df[df['user_id'].map(df['user_id'].value_counts()) >= min_posts]

    threadList = dict(df.groupby('user_id')['topic_id'].nunique())
    df['thread_count'] = df['user_id']
    df['thread_count'] = df['thread_count'].map(threadList)
    df = df[df.thread_count>=min_threads]

    return df


def make_net(forum):

    #Final ver:
    cols = 'distinct t.topic_id '
    tbl = 'topics t inner join posts p on t.topic_id = p.topic_id '
    modifier = f'WHERE forum_id = {forum} and classification2_topic >= 0.5 and length(content_post) > 10'
    #content length parameters >= 2, 5 10
    X = nx.DiGraph()

    tdf = get(tbl, cols, None, modifier)
    t = tdf['t.topic_id'].tolist()
    gdf = get('posts as p inner join topics as "t" on t.topic_id = p.topic_id', 
              'p.user_id, p.topic_id, p.post_id, p.dateadded_post', 
              f'forum_id = {forum} and classification2_topic >= 0.5 and length(content_post) > 10', None)
    # "group by p.user_id, p.post_id, p.topic_id order by p.dateadded_post asc"
    

    '''
    #Make a temporary test dataset:
    #Test Dataset Version
    X = nx.DiGraph()
    tdf = get('topics t inner join posts p on t.topic_id = p.topic_id ', 
              'distinct t.topic_id ', 
              f'forum_id = {forum} and classification2_topic >= 0.5 and length(content_post) > 10 and (t.topic_id = 7193 or t.topic_id = 7224 or t.topic_id = 7226 or t.topic_id = 7229 or t.topic_id = 7256 or t.topic_id = 7278 or t.topic_id = 7298 or t.topic_id = 7318 or t.topic_id = 7377 or t.topic_id = 7469 or t.topic_id = 7182 or t.topic_id = 7192 or t.topic_id = 7194 or t.topic_id = 7195 or t.topic_id = 7221 or t.topic_id = 7222 or t.topic_id = 7225)', None)
    t = tdf['t.topic_id'].tolist()

    gdf = get('posts as p inner join topics as t on t.topic_id = p.topic_id', 
              'p.user_id, p.topic_id, p.post_id, p.dateadded_post',
              f'forum_id = {forum} and classification2_topic >= 0.5 and length(content_post) > 10 and (t.topic_id = 7193 or t.topic_id = 7224 or t.topic_id = 7226 or t.topic_id = 7229 or t.topic_id = 7256 or t.topic_id = 7278 or t.topic_id = 7298 or t.topic_id = 7318 or t.topic_id = 7377 or t.topic_id = 7469 or t.topic_id = 7182 or t.topic_id = 7192 or t.topic_id = 7194 or t.topic_id = 7195 or t.topic_id = 7221 or t.topic_id = 7222 or t.topic_id = 7225)', None)
    '''


    for tp in range(len(t)):
        print(f'{tp+1} of {len(t)}')
        users = gdf[gdf['p.topic_id'] == t[tp]]['p.user_id'].to_numpy()

        X.add_nodes_from(users)

        for i in np.arange(0, users.size - 1):
            for j in np.arange(i + 1, users.size):
                if users[i] != users[j]:
                    if X.has_edge(users[i], users[j]):
                        X[users[i]][users[j]]['weight'] += 1
                    else:
                        X.add_edge(users[i], users[j], weight=1)
    return X

def make_net_from_df(user_data, sigma, alpha_time):
    
    X = nx.DiGraph()
    #for topic_id, group in user_data.groupby('topic_id'):
    #    display(group)
    for topic_id, group in user_data.groupby('topic_id'):
        # Using the first post date as the start date
        first_post_date = group['dateadded_post'].min()
        #print(first_post_date)
        # Filtering posts within the time window from the first post date
        #filtered_group = group[group['dateadded_post'] <= first_post_date + timedelta(days=time_window_days)]
        
        #find the root user
        #for user i and j:
            #get their delta t's
            #just plug in the formula into the weight prompts instead of 1
            #    math.e**((-(delta_t_V-delta_t_U))/sigma)

        users = group['user_id'].to_numpy()
        #is this correct with matching users / times ?
        times = group['dateadded_post'].to_numpy()
        #print(users)
        #print(times)

        X.add_nodes_from(users)
        
        for i in np.arange(0, users.size - 1):
            for j in np.arange(i + 1, users.size):
                
                #if root, then weight = 1.  we can see if it's root if delta t is 0
                if times[i] == first_post_date:
                    delta_t_VU = 0
                else:
                    #what unit of time do i make this?
                    #delta_t_V = times[j] - first_post_date
                    #delta_t_U = times[i] - first_post_date

                    #difference = (delta_t_V - delta_t_U).total_seconds()
                    delta_t_VU = (times[j]-times[i]).total_seconds()
                    delta_t_VU = delta_t_VU / (24 * 60 * 60)

                delta_t_AV = (alpha_time - times[j]).total_seconds() / (24 * 60 * 60)
                if users[i] != users[j]:
                    value = round(math.e**((-(delta_t_VU))/sigma) * math.e**((-(delta_t_AV))/sigma), 2)
                    #if value != 0:
                    #    print(value)
                    if X.has_edge(users[i], users[j]):
                        X[users[i]][users[j]]['weight'] =  round(X[users[i]][users[j]]['weight'] + value, 2)
                    else:
                        if value != 0:
                            X.add_edge(users[i], users[j], weight= value)


    return X


def save_net(N, forum_id):
    path = f"pickleX{forum_id}.p"
    with open(path, 'wb') as f:
        pk.dump(N, f)
    print(f'Pickle file saved for Forum {forum_id} at {path}...')
    return path

def get_net(path):
    with open(path, 'rb') as f:
        load = pk.load(f)
    return load

def show_net(path, forum, save = False):
    fig = plt.figure() 
    with open(path, 'rb') as f:
        load = pk.load(f)
        print('retrieved!')

    nx.draw_shell(load, with_labels = True)
    plt.show()
    if save:
        fig.savefig(f"Forum{forum} Network", dpi = 500)
    return 0

def visualize_network(net):
    '''
    with open(path, 'rb') as f:
        load = pk.load(f)
        print('retrieved!')
    '''
    plt.figure(figsize=(12, 12)) 

    pos = nx.spring_layout(net)  # the spring layout
    nx.draw(net, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    
    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(net, 'weight')
    nx.draw_networkx_edge_labels(net, pos, edge_labels=edge_labels)
    
    plt.title("Network Visualization")
    plt.show()

"""
from connect import get
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pickle as pk

def make_net(forums):

    # Query Parameters
    cols = 'distinct(topics_id) as topics'
    tbl = f'f{forums}'
    modifier = f'order by topics_id'
 
    X = nx.DiGraph()

    for forum_id in [forums]: # iterate through forums
        tdf = get(tbl, cols, None, modifier) # get all distinct topics
        t = tdf['topics'].tolist() # list of topics
        gdf = get(tbl, 'users_id, topics_id, posts_id, posted_date', None, None) # entire forum
        
        for tp in range(len(t)): #iterate through topics
            print(f'{tp+1} of {len(t)}')
            users = gdf[gdf['topics_id'] == t[tp]]['users_id'].to_numpy()

            X.add_nodes_from(users) # add nodes, duplicated nodes are ok

            for i in np.arange(0, users.size-1):
                for j in np.arange(i+1, users.size):
                    if users[i] != users[j]: #avoid cycles
                        #X.add_edge(users[i], users[j], weight = 1) # consider removing weighted edge
                        if X.has_edge(users[i], users[j]):
                            # we added this one before, just increase the weight by one
                            X[users[i]][users[j]]['weight'] += 1
                        else:
                            # new edge. add with weight=1
                            X.add_edge(users[i], users[j], weight=1)
    return X

def save_net(N, forum_id):
    path = f"pickleX{forum_id}.p"
    with open(path, 'wb') as f:
        pk.dump(N, f)
    print(f'Pickle file saved for Forum {forum_id} at {path}...')
    return path

def show_net(path, forum, save = False):
    fig = plt.figure() 
    with open(path, 'rb') as f:
        load = pk.load(f)
        print('retrieved!')

    nx.draw_shell(load, with_labels = True)
    plt.show()
    if save:
        fig.savefig(f"Forum{forum} Network", dpi = 500)
    return 0

def get_net(path):
    with open(path, 'rb') as f:
        load = pk.load(f)
        print('retrieved!')
    return load

"""
# ================================================================================================



# create driver code 
# pass forum_id into network code to construct entire network

# pass df back and iterate through
# make a code that checks forum with topics size n vs size n/2
    # equal values of each
    # bigger forum user count - data is more significant
    # more forums
# run query to retrieve topics id's where above > size and > size/2
# iterate through each topic and find first n/2 distinct users in that topic
    # feature 1
    # feature 2
    # feature 3
    # NAN, PNE, Time Elapsed, Loeveine
    # if  >= n then yes else no
    # write to dataframe or csv
        # preferably df.to_csv()


# alternate: create hierachy chart of data
    # forum --> table --> post
# create example of cascade vs noncascade
    # go thru previous papers sample graphs
# generate sample network
