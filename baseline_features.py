import pandas as pd
import networkx as nx
from sqlalchemy import create_engine
from collections import Counter
import community as community_louvain
import numpy as np
from networks import extract_forum_data, filter_length, filter_users, get_roots, get_db_connection_to_upload, upload_to_database
from sqlalchemy import create_engine, text


# Directories for outputs
#NETWORK_DIR = "networks"
#VISUALIZATION_DIR = "visualizations"
#os.makedirs(NETWORK_DIR, exist_ok=True)
#os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def load_and_filter_data(forum, minLength, minPosts, minThreads):

    forum_df = extract_forum_data(forum, minLength)
    root_users = get_roots(forum_df)
    df = filter_length(forum_df, minLength)
    df = filter_users(df, minPosts, minThreads)
    df = df.sort_values(by=['dateadded_post'])

    return df, root_users

def calculate_gini_impurity(nodes, partition):
    community_sizes = Counter(partition[node] for node in nodes if node in partition)
    total = sum(community_sizes.values())
    if total == 0:
        return 0
    return 1 - sum((size / total) ** 2 for size in community_sizes.values())

def calculate_shared_communities(partition, group1, group2):
    communities1 = set(partition[node] for node in group1 if node in partition)
    communities2 = set(partition[node] for node in group2 if node in partition)
    return len(communities1 & communities2)

def calculate_avg_time_to_adoption(timestamps, adopters):
    adoption_times = sorted(timestamps[node] for node in adopters if node in timestamps)
    if len(adoption_times) < 2:
        return 0
    total_time = sum(
        (adoption_times[i + 1] - adoption_times[i]).total_seconds()
        for i in range(len(adoption_times) - 1)
    )
    return total_time / (len(adoption_times) - 1)

def extract_features(G, adopters, lambda_frontiers, lambda_non_adopters, partition, timestamps, topic_id, group, alpha, beta, forum_id, minlength, minposts, minthreads):
    # Determine virality: topic reaches at least beta posts
    is_viral = int(len(group) >= beta)

    # Compute delta_t as the time difference between the first post and the beta-th post
    group_sorted = group.sort_values(by='dateadded_post')
    if len(group_sorted) >= beta:
        first_post_time = group_sorted.iloc[0]['dateadded_post']  # Time of the first post
        beta_post_time = group_sorted.iloc[beta-1]['dateadded_post']  # Time of the beta-th post
        delta_t_seconds = (beta_post_time - first_post_time).total_seconds()
        delta_t_hours = delta_t_seconds / 3600.0  # Convert to hours
    else:
        delta_t_hours = np.nan  # Mark as NaN if the thread doesn’t reach beta posts

    avg_time_to_adoption = calculate_avg_time_to_adoption(timestamps, adopters)

    features = {
        'topic_id': topic_id,
        'forum_id': forum_id,
        'alpha': alpha,
        'beta': beta,
        'min_post_content_length': minlength,
        'min_user_post_count': minposts,
        'min_user_thread_count': minthreads,
        'gini_impurity_adopters': calculate_gini_impurity(adopters, partition),
        'gini_impurity_lambda_frontiers': calculate_gini_impurity(lambda_frontiers, partition),
        'gini_impurity_lambda_non_adopters': calculate_gini_impurity(lambda_non_adopters, partition),
        'num_communities_adopters': len(set(partition[node] for node in adopters if node in partition)),
        'num_communities_lambda_frontiers': len(set(partition[node] for node in lambda_frontiers if node in partition)),
        'num_communities_lambda_non_adopters': len(set(partition[node] for node in lambda_non_adopters if node in partition)),
        'overlap_adopters_lambda_frontiers': calculate_shared_communities(partition, adopters, lambda_frontiers),
        'overlap_adopters_lambda_non_adopters': calculate_shared_communities(partition, adopters, lambda_non_adopters),
        'overlap_lambda_frontiers_lambda_non_adopters': calculate_shared_communities(partition, lambda_frontiers, lambda_non_adopters),
        'avg_time_to_adoption': avg_time_to_adoption / 3600.0,  # Convert to hours
        'virality': is_viral,
        'delta_t': delta_t_hours
    }
    return features


def process_all_topics_with_global_network(df, alpha, lambda_time, beta, forum_id, minlength, minposts, minthreads):
    lambda_time_delta = pd.Timedelta(hours=lambda_time)

    # Create a global graph
    global_G = nx.DiGraph()
    for topic_id, group in df.groupby('topic_id'):
        group_sorted = group.sort_values(by='dateadded_post')
        users_in_topic = group_sorted['user_id'].tolist()
        times_in_topic = group_sorted['dateadded_post'].tolist()
        for i in range(len(users_in_topic)):
            for j in range(i+1, len(users_in_topic)):
                u = users_in_topic[i]
                v = users_in_topic[j]
                global_G.add_edge(u, v, formed_at=times_in_topic[j])
    print("Completed global graph")

    # Identify cascade topics
    cascade_topics = [t_id for t_id, grp in df.groupby('topic_id') if len(grp) >= alpha]

    results = []
    count = 0
    for topic_id in cascade_topics:
        group = df[df['topic_id'] == topic_id].sort_values(by='dateadded_post')
        if len(group) < alpha:
            continue

        # Determine alpha adopters
        adopters = group['user_id'].iloc[:alpha].tolist()
        alpha_times = group['dateadded_post'].iloc[:alpha].tolist()
        adopter_to_time = dict(zip(adopters, alpha_times))

        # Determine frontiers, lambda_frontiers, non_adopters
        frontier_exposures = {}
        for adopter in adopters:
            a_time = adopter_to_time[adopter]
            if adopter in global_G:
                for successor in global_G.successors(adopter):
                    edge_data = global_G.get_edge_data(adopter, successor)
                    formed_at = edge_data['formed_at']
                    if formed_at <= a_time: #historic neighbours of alpha adopters it influences
                        if successor not in frontier_exposures or frontier_exposures[successor] > a_time:
                            frontier_exposures[successor] = a_time

        frontiers = set(frontier_exposures.keys())
        lambda_frontiers = set()
        non_adopters = set()

        timestamps = {row['user_id']: row['dateadded_post'] for _, row in group.iterrows()}
        for frontier in frontiers:
            exposure_time = frontier_exposures[frontier]
            f_post = group[group['user_id'] == frontier]
            if not f_post.empty:
                f_time = f_post['dateadded_post'].min()
                if f_time <= exposure_time + lambda_time_delta:
                    lambda_frontiers.add(frontier)
                else:
                    non_adopters.add(frontier)
            else:
                non_adopters.add(frontier)

        # Run community detection on global graph (undirected)
        partition = community_louvain.best_partition(global_G.to_undirected())
        print(f"Completed adopter and community info for topic id {topic_id}")
        features = extract_features(global_G, adopters, lambda_frontiers, non_adopters, partition, timestamps, topic_id, group, alpha, beta, forum_id, minlength, minposts, minthreads)
        results.append(features)
        count = count + 1
        print(f"Completed feature extraction for topic id {topic_id}")
        print(f"{count} / {len(cascade_topics)}")


    return pd.DataFrame(results)


def check_combination_exists(pk_values, x):

    engine = get_db_connection_to_upload()

    if x == 2:
        primary_keys = ["forum_id", "alpha", "beta", "min_post_content_length", "min_user_post_count", "min_user_thread_count"]
    elif x==1:
        primary_keys = ["forum_id", "min_post_content_length", "min_user_post_count",
                        "min_user_thread_count"]

    where_conditions = [f"{pk} = :{pk}" for pk in primary_keys]
    where_clause = " AND ".join(where_conditions)
    query = text(f"SELECT 1 FROM all_features WHERE {where_clause} LIMIT 1;")
    params = dict(zip(primary_keys, pk_values))

    with engine.connect() as conn:
        result = conn.execute(query, params).fetchone()
        return result is None  # Returns True if row does not exists, else False

if __name__ == "__main__":
    #alpha_values = [20, 10, 5]
    alpha_values = [10, 5]
    #beta_multipliers = [5,4,3,2]
    beta_multipliers = [2]
    #content_length_thresholds = [10, 2]
    content_length_thresholds = [2]
    #min_posts_values = [10, 5]
    min_posts_values = [5]

    min_threads_values = [5, 2]
    forums = [4, 8, 2, 9, 11]
    lambda_time = 24

    for forum_id in forums:
        print(f'Doing Forum {forum_id}')

        for min_post_length in content_length_thresholds:
            print(f"Min content length filter = {min_post_length}")

            for min_posts_per_user in min_posts_values:
                print(f'User post minimum: {min_posts_per_user}')

                for min_threads_per_user in min_threads_values:
                    print(f'User thread minimum: {min_threads_per_user}')

                    if(check_combination_exists([forum_id, min_post_length,min_posts_per_user,min_threads_per_user],1)):
                        print(f"Combination of f = {forum_id}, post length = {min_post_length}, post count = {min_posts_per_user}, thread count = {min_threads_per_user} DNE")
                        continue

                    df, roots = load_and_filter_data(forum_id, min_post_length, min_posts_per_user,
                                                     min_threads_per_user)

                    for alpha in alpha_values:
                        print(f'\nDoing alpha {alpha}')

                        for m in beta_multipliers:
                            beta = alpha * m
                            print(f'Beta multiplier {m}, beta = {beta}')
                            if (check_combination_exists([forum_id, alpha, beta, min_post_length, min_posts_per_user, min_threads_per_user], 2)):
                                print(f"Combination of f = {forum_id}, a = {alpha}, b = {beta}, post length = {min_post_length}, post count = {min_posts_per_user}, thread count = {min_threads_per_user} DNE")
                                continue

                            cascade_features = process_all_topics_with_global_network(df, alpha, lambda_time, beta, forum_id, min_post_length, min_posts_per_user, min_threads_per_user)
                            print("Feature Extraction Complete:")
                            upload_to_database(cascade_features, 'baseline_features')
                            print("Features saved")
                            del cascade_features
                        beta_multipliers = [5, 4, 3, 2]

                    alpha_values = [20, 10, 5]

            min_posts_values = [10, 5]

        content_length_thresholds = [10, 2]