
from connect import get
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pickle as pk
import datetime as dt
import community
from networks import make_net_from_df, visualize_network
from IPython.display import display


X = None
C = None
UX = None
IX = None
PR = None
B = None
K = None
partition = None
P = None

def test(network):
    return nx.strongly_connected_components(network)

def test2(network):
    return nx.weakly_connected_components(network)

def remove_small_components(G, min_size):
    # Find all connected components
    connected_components = list(nx.strongly_connected_components(G))

    # Create a new graph to store large components
    large_components_graph = nx.Graph()

    # Add nodes and edges from components larger than min_size
    for component in connected_components:
        if len(component) >= min_size:
            large_components_graph.add_nodes_from(component)
            # Add edges between nodes in the same component
            subgraph = G.subgraph(component)
            for u, v in subgraph.edges():
                large_components_graph.add_edge(u, v)

    return large_components_graph

#INNOVATOR FEATURES
def root_f1(root): # root user out_degree centrality / 
    return nx.out_degree_centrality(X)[root]

def root_f2(root): # root user eigenvector centrality
    # we want to replace eigenvector centrality with page rank, as it is more suitable for our data
    return C[root]

def root_f2_2(root): #page rank
    return P[root]

#Betweenness centrality
def root_f3(root):
    return B[root]

#Closeness centrality
def root_f4(root): # closeness centrality 
    return nx.closeness_centrality(IX.reverse(), root, distance='weight') #needs inverse weight

#K-shell
def root_f5(root):
    return K[root]

#EARLY ADOPTER FEATURES
#for all early adopters, maybe create a subgraph so that other nodes outside are not included in any calculations


def early_f1(users): # G.out_degree(1) avg
    sum = 0
    for usr in users:
        sum += (nx.out_degree_centrality(X)[usr])
    return sum/len(users)

def early_f2(users): # user eigenvector centrality
    sum = 0
    for usr in users:
        sum += C[usr]
    return sum/len(users)

def early_f2_2(users): #page rank
    sum = 0
    for usr in users:
        sum += P[usr]
    return sum/len(users)

def early_f3(users):
    sum = 0
    for usr in users:
        sum += B[usr]
    return sum/len(users)

def early_f4(users): # closeness centrality
    sum = 0
    for usr in users:
        sum +=  nx.closeness_centrality(IX.reverse(), usr, distance='weight')
    return sum/len(users)

def early_f5(users):
    sum = 0
    for usr in users:
        sum += K[usr]
    return sum/len(users)


def forward_neighbors(graph, current_n, backward_n):

    foward_neighbor_weights = dict()

    # Iterate over each node in the group of nodes
    for node in current_n:
        # Iterate over the neighbors of each node
        for neighbor in graph.neighbors(node):
            # Exclude the nodes in the original group
            if neighbor not in current_n and neighbor not in backward_n:
                # Sum the weights to each neighbor
                if neighbor in foward_neighbor_weights:
                    foward_neighbor_weights[neighbor] += graph[node][neighbor].get('weight')
                else:
                    foward_neighbor_weights[neighbor] = graph[node][neighbor].get('weight')

    return foward_neighbor_weights

def early_f6_f7_nexthop(users): #Average out-degree weight to 1-Hop forward neighbors (averaged by neighborhood size)
    result = forward_neighbors(X, users, set())
    forward_n = set(result.keys())
    forward_n_size = len(forward_n)
    #avg_forward_weight = np.median(list(result.values()))
    avg_forward_weight = sum(result.values()) / forward_n_size if forward_n_size != 0 else 0
    #print(f"Median:{avg_forward_weight}")
    #print(f"mean:{np.mean(list(result.values()))}")
    return avg_forward_weight, forward_n_size, forward_n


def get_components(subgraph):
    components = list(nx.strongly_connected_components(subgraph))
    return components #for f8, just do len() of this value

def early_f9(subgraph, components):
    graph_order = subgraph.order()
    diameter_score = 0
    max_d = 0
    for component in components:
        sub = subgraph.subgraph(component)
        sub_order = sub.order()
        result = nx.algorithms.distance_measures.diameter(sub, weight='weight')
        if result > max_d:
            max_d = result
        #print(f"{result} / {sub_order / graph_order} = {result / (sub_order / graph_order) if result != 0 else 1}")
        diameter_score += result / (sub_order / graph_order) if result != 0 else 1
    #return diameter_score, max_d
    return max_d


def early_f10(subgraph): 
    return nx.density(subgraph) 

def early_f11(subgraph, components):
    graph_order = subgraph.order()
    aspl_score = 0
    max_aspl = 0
    for component in components:
        sub = subgraph.subgraph(component)
        sub_order = sub.order()
        result = nx.average_shortest_path_length(sub, weight='weight')
        if result > max_aspl:
            max_aspl = result
        #print(f"{result} / {sub_order / graph_order} = {result / (sub_order / graph_order) if result != 0 else 1}")
        aspl_score += result / (sub_order / graph_order) if result != 0 else 1
    #return aspl_score, max_aspl
    return max_aspl

def early_f12(subgraph):
    return nx.average_clustering(subgraph, weight='weight')
    #use X or a subgraph of users?

def early_f13(users): #hub count
    #not sure how to implement this, used chatgpt for a base
    degree_centrality = nx.degree_centrality(X)
    subset_centrality = {node: degree_centrality[node] for node in users if node in degree_centrality}

    #hub threshold = ?
    hub_threshold = 0.1
    hubs = {node: score for node, score in subset_centrality.items() if score > hub_threshold}
    return len(hubs)

def gini_impurity(intersection, size): #gini impurity
    if size==0: return 0
    return 1 - sum((x / size) ** 2 for x in intersection)

def early_f14_15_16(users, one_hop):     # num of communities, gini impurity
    #might need some review 
    # make subgraph
    #S = UX.subgraph(users) # undirected
    # j = len([users for _, users in partition.items() if users != 0])
    j = len(set([comm for node, comm in partition.items() if node in users]))
    #modularity = community.modularity(partition, S, weight='weight')

    gini = early_community_gini(partition, users)
    shared = early_community_shared(partition, users, one_hop)
    return j, gini, len(shared)

def early_community_gini(partition, users): #gini impurity of the communities
    community_sizes = {}
    for node, comm in partition.items():
        if node in users:
            community_sizes[comm] = community_sizes.get(comm, 0) + 1
        # if comm in community_sizes.keys():
        #     community_sizes[comm]['total'] = community_sizes[comm]['total'] + 1
        # else:
        #     community_sizes[comm] = {'total': 1, 'users': 0}
        # if node in users:
        #     community_sizes[comm]['users'] = community_sizes[comm]['users'] + 1

    sizes = list(community_sizes.values())
    return gini_impurity(sizes, len(users))

def early_community_shared(partition, early_adopt, one_hop): #shared communities with 1 hop adopters
    early_adopter_communities = set()
    one_hop_frontier_communities = set()

    # Collect communities for Early Adopters
    for node in early_adopt:
        if node in partition:
            early_adopter_communities.add(partition[node])

    # Collect communities for 1-Hop Frontiers
    for node in one_hop:
        if node in partition:
            one_hop_frontier_communities.add(partition[node])
    
    return early_adopter_communities.intersection(one_hop_frontier_communities)


def early_f17(times): # average time to adoption
    sum = dt.timedelta(days=0)
    for i in range(len(times)-1):
        sum = sum + times[i+1] - times[i] 
    return round(sum.total_seconds()/(60*60*24*(len(times)-1)),2) # avg and in minutes

#def early_alpha_time():
def early_f18(pst_tm): # time elapsed
    elapsed = pst_tm[-1] - pst_tm[0]
    return round(elapsed.total_seconds()/(60*60*24),2)

#...

#Old Dhanush Functions that are not used for now:
'''
def get_f1(users): # sum of number of neighbors
    sum = 0
    for usr in users:
        sum += len(list(X.neighbors(usr)))
    #return sum/len(users)
    return sum

def get_f2(root): # NAN for root
    #probably not included because we have out-degree centrality already
    return len(list(X.neighbors(root)))

def get_f4(pst_tm): # time elapsed
    elapsed = pst_tm[-1] - pst_tm[0]
    return round(elapsed.total_seconds()/60,2)

def get_f5(root): # root user degree centrality --> edges/neighbors
    return nx.degree_centrality(X)[root]

def get_f8(root): # cumulative weight of out degree edges
    return X.out_degree(weight = 'weight')[root]

def get_f9(users): # average cumulative weight of out_degree edges
    sum = 0
    for usr in users:
        sum += X.out_degree(weight = 'weight')[usr]
    return sum/len(users)

def get_f10(root): # root user pagerank --> importance ranking
    return PR[root]

def get_f11(users): # average page rank
    sum = 0
    for usr in users:
        sum += PR[usr]
    return sum/len(users)

#how are the group functions different the other ones? understand these group functions

def get_f12(users): # group out degree centrality
    return nx.group_out_degree_centrality(X, users)

def get_f13(users): # group_betweenness centrality
    return nx.group_betweenness_centrality(X, users, normalized=True, weight='weight')

def get_f14(users): # group closeness centrality - measure of how close the group is to the other nodes in the graph.
    return nx.group_closeness_centrality(X, users, weight='weight')

def get_f15(times): # average time to adoption
    sum = dt.timedelta(days=0)
    for i in range(len(times)-1):
        sum = sum + times[i+1] - times[i] 
    return round(sum.total_seconds()/(60*(len(times)-1)),2) # avg and in minutes

def get_f5(users): # gini coeffient https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
    return sum/len(users)

def get_f6(users): # louveine https://python-louvain.readthedocs.io/en/latest/
    return sum/len(users)

def get_communities(users): # num of communities, modularity
    # make subgraph
    S = UX.subgraph(users) # undirected
    lp = community.best_partition(S, weight='weight', random_state=40)
    j = len([users for _, users in lp.items() if users != 0])
    mod = community.modularity(lp, S, weight='weight')
    return j, mod
'''

def make_subgraph(G, users):
    H = G.subgraph(users)
    #pos = nx.spring_layout(H, scale=30, k=7/np.sqrt(H.order()))
    #nx.draw(H, pos, node_color='red', 
    #with_labels=True)
    #plt.show()
    return H

#get_features(train_threads, train_times, get_net(pkp))

def get_features(csc, ncsc, tmcsc, tmncsc, df, sigma, filters):
    data = [] # topic_id f1, f2, f3, f4, yes
    forumNum = filters[0]
    alpha = filters[1]
    beta = filters[2]
    minLength = filters[3]
    minPosts = filters[4]
    minThreads = filters[5]
    '''
    global X    
    #X = make_net_from_df(df) # entire network
    
    global UX
    UX = X.to_undirected()

    global IX
    IX = X.copy()
    for u,v,d in IX.edges(data=True):
        d['weight'] = 1/d['weight']

    global partition
    partition = community.best_partition(UX, weight='weight', random_state=40)

    
    strongs = list(test(X))
    for i, x in enumerate(strongs):
        print(i, len(x))
    X = remove_small_components(X, 5)
    strongs = list(nx.connected_components(X))
    for i, x in enumerate(strongs):
        print(i, len(x))   
    
    testval = test(X)
    for component in testval:
        print(component)
    testval = test2(X)
    for component in testval:
        print(component)
    
    
    global C 
    #C = nx.eigenvector_centrality_numpy(X.reverse(), weight='weight') 
    C = nx.pagerank(X.reverse(), weight='weight')

    # centrality of all nodes
    # have to reverse graph for out-edges eigenvector centrality

    global B 
    B = nx.betweenness_centrality(IX, weight='weight')

    global K
    K = nx.core_number(X)
    

    global PR 
    PR = nx.pagerank_numpy(X, alpha=0.9, weight = 'weight')
    '''
    
    #keyerror = pos
    #csc = threads['Pos']
    #ncsc = threads['Neg']
    #tmcsc = times['Pos']
    #tmncsc = times['Neg']

    print('Doing Yes Cases')
    for key, users in csc.items():
        print(f'Doing Thread {key}')
        print(users)

        alpha_time = tmcsc[key][-1]
        print(alpha_time)
        alpha_dataframe = df[df['dateadded_post'] <= alpha_time]
        #display(alpha_dataframe)
        
        global X
        X = make_net_from_df(alpha_dataframe, sigma, alpha_time)
        #visualize_network(X)
        global UX
        UX = X.to_undirected()

        global IX
        IX = X.copy()
        for u,v,d in IX.edges(data=True):
            d['weight'] = 1/d['weight']

        global partition
        partition = community.best_partition(UX, weight='weight', random_state=40)

        global C 
        C = nx.eigenvector_centrality_numpy(X.reverse(), weight='weight')

        global P
        P = nx.pagerank(X.reverse(), weight='weight')

        # centrality of all nodes
        # have to reverse graph for out-edges eigenvector centrality

        global B 
        B = nx.betweenness_centrality(IX, weight='weight')

        global K
        K = nx.core_number(X)

        root = users[0]
 
        rf1 = root_f1(root)
        rf2 = root_f2(root)
        rf2_2 = root_f2_2(root)
        rf3 = root_f3(root)
        rf4 = root_f4(root)
        rf5 = root_f5(root)
 
         #centrality
        ef1 = early_f1(users)
        ef2 = early_f2(users)
        ef2_2 = early_f2_2(users)
        ef3 = early_f3(users)
        ef4 = early_f4(users)
        ef5 = early_f5(users)

        #forward-connectivity
        ef6, ef7, one_hop = early_f6_f7_nexthop(users)

        #network-based
        early_subgraph = make_subgraph(X, users)
        early_subgraph_inverse = make_subgraph(IX, users)
        early_components = get_components(early_subgraph_inverse) #inverse or not? does it matter?
        ef8 = len(early_components)
        ef9 = early_f9(early_subgraph_inverse, early_components) 
        ef10 = early_f10(early_subgraph)
        ef11 = early_f11(early_subgraph_inverse, early_components) 
        
        ef12 = early_f12(early_subgraph)
        ef13 = early_f13(users)
        #print(ef12)
        #print(ef13)

        #community
        ef14, ef15, ef16 = early_f14_15_16(users, one_hop) 

        #temporal
        ef17 = early_f17(tmcsc[key])
        ef18 = early_f18(tmcsc[key])

        data.append([key,forumNum,alpha,beta,minLength,minPosts,minThreads,sigma, rf1,rf2,rf2_2,rf3,rf4,rf5, ef1,ef2,ef2_2,ef3,ef4,ef5,ef6,ef7,ef8,ef9,ef10,ef11,ef12,ef13,ef14,ef15,ef16,ef17,ef18, 1]) # topic_id f1, f2, f3, f4... no

    print('Doing No Cases')
    for key, users in ncsc.items():
        print(f'Doing Thread {key}')
        print(users)
        alpha_time = tmncsc[key][-1]
        print(alpha_time)
        alpha_dataframe = df[df['dateadded_post'] <= alpha_time]
        #display(alpha_dataframe)
        X = make_net_from_df(alpha_dataframe, sigma, alpha_time)

        UX = X.to_undirected()

        IX = X.copy()
        for u,v,d in IX.edges(data=True):
            d['weight'] = 1/d['weight']

        partition = community.best_partition(UX, weight='weight', random_state=40)

        C = nx.eigenvector_centrality_numpy(X.reverse(), weight='weight') 
        
        P = nx.pagerank(X.reverse(), weight='weight')

        # centrality of all nodes
        # have to reverse graph for out-edges eigenvector centrality

        B = nx.betweenness_centrality(IX, weight='weight')

        K = nx.core_number(X)


        root = users[0]
        rf1 = root_f1(root)
        rf2 = root_f2(root)
        rf2_2 = root_f2_2(root)
        rf3 = root_f3(root)
        rf4 = root_f4(root)
        rf5 = root_f5(root)
 
         #centrality
        ef1 = early_f1(users)
        ef2 = early_f2(users)
        ef2_2 = early_f2_2(users)
        ef3 = early_f3(users)
        ef4 = early_f4(users)
        ef5 = early_f5(users)

        #forward-connectivity
        ef6, ef7, one_hop = early_f6_f7_nexthop(users)

        #network-based
        early_subgraph = make_subgraph(X, users)
        early_subgraph_inverse = make_subgraph(IX, users)
        early_components = get_components(early_subgraph_inverse) #inverse or not? does it matter?
        ef8 = len(early_components)
        ef9 = early_f9(early_subgraph_inverse, early_components) 
        ef10 = early_f10(early_subgraph)
        ef11 = early_f11(early_subgraph_inverse, early_components) 
        ef12 = early_f12(early_subgraph)
        ef13 = early_f13(early_subgraph)

        #community
        ef14, ef15, ef16 = early_f14_15_16(users, one_hop) 

        #temporal
        ef17 = early_f17(tmncsc[key])
        ef18 = early_f18(tmncsc[key])


        data.append([key,forumNum,alpha,beta,minLength,minPosts,minThreads,sigma, rf1,rf2,rf2_2,rf3,rf4,rf5, ef1,ef2,ef2_2,ef3,ef4,ef5,ef6,ef7,ef8,ef9,ef10,ef11,ef12,ef13,ef14,ef15,ef16,ef17,ef18, 0]) # topic_id f1, f2, f3, f4... no

    pdf = pd.DataFrame(data, columns=['Topic', 'Forum', 'Alpha', 'Beta', 'Min Post Content Length',
                                      'Min User Post Count', 'Min User Thread Count', 'Sigma',
                                      'RF1', 'RF2', 'RF2.2', 'RF3', 'RF4', 'RF5', 'EF1', 'EF2', 'EF2.2', 'EF3', 'EF4', 'EF5', 
                                      'EF6', 'EF7', 'EF8', 'EF9', 'EF10', 'EF11', 'EF12', 'EF13',
                                      'EF14', 'EF15', 'EF16', 'EF17', 'EF18', 'Class'])
    return pdf


'''
References
    ----------
    .. [1] Phillip Bonacich:
       Power and Centrality: A Family of Measures.
       American Journal of Sociology 92(5):1170–1182, 1986
       http://www.leonidzhukov.net/hse/2014/socialnetworks/papers/Bonacich-Centrality.pdf
    .. [2] Mark E. J. Newman:
       Networks: An Introduction.
       Oxford University Press, USA, 2010, pp. 169.
    """
    '''