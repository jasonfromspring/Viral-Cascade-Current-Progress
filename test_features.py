
from connect import get
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pickle as pk
import datetime as dt
import community
import random

X = None
C = None
UX = None
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
    return nx.closeness_centrality(X.reverse(), root, distance='weight') #needs inverse weight

#K-shell
def root_f5(root):
    return K[root]

#EARLY ADOPTER FEATURES
#for all early adopters, maybe create a subgraph so that other nodes outside are not included in any calculations


def early_f1(users): # G.out_degree(1) avg
    sum = 0
    for usr in users:
        sum += (X.out_degree(usr))
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
        sum +=  nx.closeness_centrality(X.reverse(), usr, distance='weight')
    return sum/len(users)

def early_f5(users):
    sum = 0
    for usr in users:
        sum += K[usr]
    return sum/len(users)

def early_f8(subgraph):
    return nx.diameter(subgraph, weight='weight') #might need fix, make subgraph of only early adopters first

def early_f9(subgraph): 
    return nx.density(subgraph)  #might need fix, make subgraph of only early adopters first

def early_f10(subgraph):
    return nx.average_shortest_path_length(subgraph, weight='weight') #see above

def early_f11(subgraph):
    return nx.clustering(subgraph, weight='weight') #see above, or set nodes = [wanted nodes]
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
    
    # make subgraph
    S = UX.subgraph(users) # undirected
    partition = community.best_partition(S, weight='weight', random_state=40)
    j = len([users for _, users in partition.items() if users != 0])
    modularity = community.modularity(partition, S, weight='weight')

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

def make_subgraph(users):
    H = X.subgraph(users)
    pos = nx.spring_layout(H, scale=30, k=7/np.sqrt(H.order()))
    nx.draw(H, pos, node_color='red', 
    with_labels=True)
    plt.show()
    return H

#get_features(train_threads, train_times, get_net(pkp))
def get_features(network, early_adopters, early_times, one_hop):
    data = [] # topic_id f1, f2, f3, f4, yes
    
    global X    
    X = network # entire network
    
    global UX
    UX = X.to_undirected()

    strongs = list(test(X))
    for i, x in enumerate(strongs):
        print(i, len(x))

    global C 
    C = nx.eigenvector_centrality_numpy(X.reverse(), weight='weight') 
    # centrality of all nodes
    # have to reverse graph for out-edges eigenvector centrality

    global B 
    B = nx.betweenness_centrality(X, weight='weight')

    global K
    K = nx.core_number(X)

    '''global PR 
    PR = nx.pagerank_numpy(X, alpha=0.9, weight = 'weight')'''
    root = early_adopters[0]
    rf1 = root_f1(root)
    rf2= root_f2(root)
    rf3 = root_f3(root)
    rf4= root_f4(root)
    rf5 = root_f5(root)

    print("Innovator")
    print(rf1, rf2, rf3, rf4, rf5)

    #centrality
    ef1 = early_f1(early_adopters)
    ef2 = early_f2(early_adopters)
    ef3 = early_f3(early_adopters)
    ef4 = early_f4(early_adopters)
    ef5 = early_f5(early_adopters)

    print(f"\nEarly Adopters")
    print(ef1, ef2, ef3, ef4, ef5)
    #forward-connectivity
    #ef6 = early_f6(early_adopters)
    #ef7 = early_f7(early_adopters)


    #network-based
    early_subgraph = make_subgraph(early_adopters)
    ef8 = early_f8(early_subgraph) #has error
    ef9 = early_f9(early_subgraph)
    ef10 = early_f10(early_subgraph) #has error
    ef11 = early_f11(early_subgraph)
    ef12 = early_f12(early_subgraph)
    print(ef8, ef9, ef10, ef11, ef12)

    #community
    ef13, ef14, ef15 = early_f13_14_15(early_adopters, one_hop) 
    print(ef13, ef14, ef15)

    #temporal
    ef16 = early_f16(early_times)
    ef17 = early_f17(early_times)
    print(ef16, ef17)

# Create a directed graph
G = nx.DiGraph()
"""
# Add 10 people (nodes) with names A to J
nodes = [chr(i) for i in range(ord('A'), ord('A') + 10)]

G.add_nodes_from(nodes)

additional_nodes = [chr(i) for i in range(ord('K'), ord('K') + 10)]
G.add_nodes_from(additional_nodes)

# Define directed edges with set weights
edges = [
    ('A', 'B', 3), ('B', 'C', 2), ('C', 'D', 5), ('D', 'E', 1),
    ('E', 'F', 4), ('F', 'G', 3), ('G', 'H', 2), ('H', 'A', 4), # Cycle among A to H
    ('A', 'I', 1), ('B', 'J', 2), ('C', 'I', 3), ('D', 'J', 4), # Connections to I and J
    ('I', 'E', 2), ('J', 'F', 1), ('G', 'I', 5), ('H', 'J', 3), # Connections back to A-H
    ('I', 'J', 4), ('J', 'I', 2) # Strong connection between I and J
]

# Add directed edges with weights to the graph
G.add_weighted_edges_from(edges)



# Add sparse or isolated edges for nodes K to T (not strongly connected)
edges_sparse = [
    ('K', 'L', 1), ('L', 'M', 3), ('N', 'O', 2), ('O', 'P', 4),
    ('P', 'Q', 5), ('Q', 'R', 1), ('S', 'T', 3), 
    ('J', 'K', 2), ('J', 'L', 3), ('J', 'M', 1)
]

# Add sparse edges for nodes K to T
G.add_weighted_edges_from(edges_sparse)

# Draw the directed network with edge labels
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700, font_size=10, font_color="black", edge_color="gray", arrows=True)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
plt.title("Directed Network of 10 People with Set Edge Weights")
plt.show()

#strongs = list(test(G))
#for i, x in enumerate(strongs):
#    print(i, len(G))

num_edges = 24
start_date = dt.datetime(2023, 1, 1, 8, 0)  # Starting at 8 AM on January 1, 2023
time_interval = dt.timedelta(hours=1)        # 1-hour interval between each datetime

# Generate the list of datetimes
times = [start_date + i * time_interval for i in range(num_edges)]
"""

nodes = ['A','B','C']
G.add_nodes_from(nodes)
additional_nodes = ['D','E','F']
G.add_nodes_from(additional_nodes)
edges = [
    ('A','B',1), ('B','C',1), ('A','C',1),
    ('B','A',1), ('C','B',1),('C','A',1),

    ('C', 'D', 1), ('B','D',1), ('A','D',1),
    ('D','C',1), ('D','B',1), ('D','A',1),
    ('D', 'E', 1), ('E', 'F', 1), ('F','E', 1)
]
G.add_weighted_edges_from(edges)

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700, font_size=10, font_color="black", edge_color="gray", arrows=True)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
plt.title("Directed Network of 10 People with Set Edge Weights")
plt.show()


num_edges = 15
start_date = dt.datetime(2023, 1, 1, 8, 0)  # Starting at 8 AM on January 1, 2023
time_interval = dt.timedelta(hours=1)        # 1-hour interval between each datetime

# Generate the list of datetimes
times = [start_date + i * time_interval for i in range(num_edges)]

get_features(G, nodes, times, additional_nodes)
