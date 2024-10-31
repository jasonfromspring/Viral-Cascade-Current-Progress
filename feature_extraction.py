
from connect import get
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pickle as pk
import datetime as dt
import community

X = None
C = None
UX = None
IX = None
PR = None
B = None
K = None

#check the number of connected components
#nx.number_strongly_connected_components(G)
#nx.number_weakly_connected_components(G)
#will decide if we normalize certain features or not

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

def early_f8(subgraph):
    return nx.diameter(subgraph, weight='weight')

def early_f9(subgraph): 
    return nx.density(subgraph) 

def early_f10(subgraph):
    return nx.average_shortest_path_length(subgraph, weight='weight') 

def early_f11(subgraph):
    return nx.clustering(subgraph, weight='weight')
    #there is an average clustering networkx function, should we use that instead?

def early_f12(users): #hub count
    #not sure how to implement this, used chatgpt for a base
    degree_centrality = nx.degree_centrality(X)
    subset_centrality = {node: degree_centrality[node] for node in users if node in degree_centrality}

    #hub threshold = ?
    hub_threshold = 0.1
    hubs = {node: score for node, score in subset_centrality.items() if score > hub_threshold}
    return len(hubs)

def gini_impurity(sizes): #gini impurity
    n = sum(sizes)
    if n==0: return 0
    return 1 - sum((size / n) ** 2 for size in sizes)

def early_f13_14_15(users, one_hop):     # num of communities, gini impurity
    #might need some review 
    # make subgraph
    S = UX.subgraph(users) # undirected
    partition = community.best_partition(S, weight='weight', random_state=40)
    j = len([users for _, users in partition.items() if users != 0])
    #modularity = community.modularity(partition, S, weight='weight')

    gini = early_community_gini(partition)
    shared = early_community_shared(partition, users, one_hop)
    return j, gini, shared

def early_community_gini(partition): #gini impurity of the communities
    community_sizes = {}
    for node, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    sizes = list(community_sizes.values())
    return gini_impurity(sizes)

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


def early_f16(times): # average time to adoption
    sum = dt.timedelta(days=0)
    for i in range(len(times)-1):
        sum = sum + times[i+1] - times[i] 
    return round(sum.total_seconds()/(60*(len(times)-1)),2) # avg and in minutes

#def early_alpha_time():
def early_f17(pst_tm): # time elapsed
    elapsed = pst_tm[-1] - pst_tm[0]
    return round(elapsed.total_seconds()/60,2)

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
    pos = nx.spring_layout(H, scale=30, k=7/np.sqrt(H.order()))
    nx.draw(H, pos, node_color='red', 
    with_labels=True)
    plt.show()
    return H

#get_features(train_threads, train_times, get_net(pkp))
def get_features(threads, times, network):
    data = [] # topic_id f1, f2, f3, f4, yes
    
    global X    
    X = network # entire network
    
    global UX
    UX = X.to_undirected()

    global IX
    IX = X.copy()
    for u,v,d in IX.edges(data=True):
        d['weight'] = 1/d['weight']

    
    '''
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
    '''
    global C 
    C = nx.eigenvector_centrality_numpy(X.reverse(), weight='weight') 
    # centrality of all nodes
    # have to reverse graph for out-edges eigenvector centrality

    global B 
    B = nx.betweenness_centrality(IX, weight='weight')

    global K
    K = nx.core_number(X)

    '''global PR 
    PR = nx.pagerank_numpy(X, alpha=0.9, weight = 'weight')'''
    
    #keyerror = pos
    csc = threads['Pos']
    ncsc = threads['Neg']
    tmcsc = times['Pos']
    tmncsc = times['Neg']

    print('Doing Yes Cases')
    for key, users in csc.items():
        #original code said doing "forum" instead of "thread", may be a mistake?
        print(f'Doing Thread {key}')
        root = users[0]
        f1 = root_f1(root)
        f2 = root_f2(root)
        f3 = root_f3(root)
        f4 = root_f4(root)
        f5 = root_f5(root)
 
         #centrality
        ef1 = early_f1(users)
        ef2 = early_f2(users)
        ef3 = early_f3(users)
        ef4 = early_f4(users)
        ef5 = early_f5(users)

        #forward-connectivity
        #ef6 = early_f6(early_adopters)
        #ef7 = early_f7(early_adopters)

        #network-based
        early_subgraph = make_subgraph(X, users)
        early_subgraph_inverse = make_subgraph(IX, users)
        ef8 = early_f8(early_subgraph_inverse) 
        ef9 = early_f9(early_subgraph)
        ef10 = early_f10(early_subgraph_inverse) 
        ef11 = early_f11(early_subgraph)
        ef12 = early_f12(early_subgraph)

        #community
        #ef13, ef14, ef15 = early_f13_14_15(users, one_hop) 

        #temporal
        ef16 = early_f16(tmcsc[key])
        ef17 = early_f17(tmcsc[key])


        data.append([f'Topic{key}',f1,f2,f3,f4,f5, ef1, ef2, ef3, ef4, ef5, ef8, ef9, ef10, ef11, ef12, ef16, ef17, 1]) # topic_id f1, f2, f3, f4... yes
        #make_subgraph(value)


    print('Doing No Cases')
    for key, users in ncsc.items():
        #same ? as csc
        print(f'Doing Thread {key}')

        root = users[0]
        f1 = root_f1(root)
        f2 = root_f2(root)
        f3 = root_f3(root)
        f4 = root_f4(root)
        f5 = root_f5(root)
 
         #centrality
        ef1 = early_f1(users)
        ef2 = early_f2(users)
        ef3 = early_f3(users)
        ef4 = early_f4(users)
        ef5 = early_f5(users)

        #forward-connectivity
        #ef6 = early_f6(early_adopters)
        #ef7 = early_f7(early_adopters)

        #network-based
        early_subgraph = make_subgraph(X, users)
        early_subgraph_inverse = make_subgraph(IX, users)
        ef8 = early_f8(early_subgraph_inverse) 
        ef9 = early_f9(early_subgraph)
        ef10 = early_f10(early_subgraph_inverse) 
        ef11 = early_f11(early_subgraph)
        ef12 = early_f12(early_subgraph)

        #community
        #ef13, ef14, ef15 = early_f13_14_15(users, one_hop) 

        #temporal
        ef16 = early_f16(tmcsc[key])
        ef17 = early_f17(tmcsc[key])


        data.append([f'Topic{key}',f1,f2,f3,f4,f5, ef1, ef2, ef3, ef4, ef5, ef8, ef9, ef10, ef11, ef12, ef16, ef17, 0]) # topic_id f1, f2, f3, f4... yes

    pdf = pd.DataFrame(data, columns=['Topic', 'F1', 'F2', 'F3', 'F4', 'F5', 'EF1', 'EF2', 'EF3,' 'EF4', 'EF5,' 'EF8',
                                        'EF9', 'EF10', 'EF11', 'EF12', 'EF16', 'EF17' 'Class'])
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