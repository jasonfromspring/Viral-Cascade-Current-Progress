import datetime as dt
import pandas as pd
from IPython.display import display
import numpy as np

def get_avg_time_difference(df):
    topics = df['topic_id'].unique()
    pos = []
    x = 0
    y = 0
    data = []
    for topic in topics:

        time_df = df[df['topic_id']==topic]
        times = time_df['dateadded_post'].tolist()

        sum = dt.timedelta(days=0)
        diff_count = len(times)-1

        if diff_count != 0:
            for i in range(len(times)-1):
                diff = times[i+1] - times[i]
                sum = sum + diff
            sum = sum.total_seconds() / (24*60*60)
            avg = sum / (len(times)-1)
            pos.append(avg)
            x += 1
            data.append([topic, avg])
        else:
            y += 1
    
    pdf = pd.DataFrame(data, columns = ['Topic ID', 'Average Time'])
    display(pdf)
    print("Total")

    print(f"Mean: {np.mean(pos)}")
    print(f"Total number of threads with more than 1 post: {x}")
    print(f"Total number of threads with only one post: {y}")
    print(f"Median; {np.median(pos)}")

def average_time_everything(df):
    
    topics = df['topic_id'].unique()
    data = []
    all_averages = []
    x = 0
    y = 0
    differences = []
    for topic in topics:

        time_df = df[df['topic_id']==topic]
        times = time_df['dateadded_post'].to_numpy()

        first_post_date = np.min(times)
        sum = 0      
        if times.size - 1 != 0:
            count = 0
            for i in np.arange(0, times.size - 1):
                for j in np.arange(i + 1, times.size):
                    
                    deltaT = (times[j] - times[i]).total_seconds()
                    deltaT = deltaT / (24*60*60)
                    differences.append(deltaT)
                    sum += deltaT
                    count += 1
                
            avg = sum / count
            all_averages.append(avg)
            data.append([topic, avg])
            x += 1
        else:
            y += 1

    pdf = pd.DataFrame(data, columns = ['Topic ID', 'Average Time'])
    display(pdf)
    print("Total")

    print(f"Mean: {np.mean(all_averages)}")
    print(f"Total number of threads with more than 1 post: {x}")
    print(f"Total number of threads with only one post: {y}")
    print(f"Median; {np.median(all_averages)}")
    print(f"Super avg: {np.mean(differences)}")
    print(f"Super med: {np.median(differences)}")
