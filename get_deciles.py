import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# Path to the directory containing JSON files
directory = "data"

def get_user_activity(file_path):

    # Initialize dictionary to store author submissions
    author_submissions = {}

    # Read JSON file
    with open(file_path) as file:
        entries = file.readlines()

    # Process each entry in the JSON file
    for entry in entries:
        try:
            data = json.loads(entry)

            # Filter entries based on created_utc and author conditions
            if (
                "author" in data
                and "created_utc" in data
                and data["author"] != "[deleted]"
                and data["author"] != "AutoModerator"
                and int(data["created_utc"]) > 1546300801
            ):
                author = data["author"]

                # Increment author's submission count
                if author in author_submissions:
                    author_submissions[author] += 1
                else:
                    author_submissions[author] = 1
        except json.JSONDecodeError:
            print("Error decoding JSON entry:", entry)
            continue

    # Convert author submissions dictionary to a DataFrame
    df = pd.DataFrame(list(author_submissions.items()), columns=["Author", "Submissions"])
    return df

def calc_numiles(df, num_numiles):
    # Numiles
    activity_counts = df["Submissions"].value_counts().sort_index()
    deciles = np.percentile(activity_counts, range(0, 99, 100//num_numiles))
    return deciles.tolist()

def get_numiles(file_path, num_numiles):
    df = get_user_activity(file_path)
    numiles = calc_numiles(df, num_numiles)
    return numiles

def partition_users(df, numiles):
    # Returns a dictionary of lists of the form {numile: [user1, user2, ...]}
    partition = {}
    for i in range(len(numiles) - 1):
        partition[i] = df[(df["Submissions"] >= numiles[i]) & (df["Submissions"] < numiles[i+1])]["Author"].tolist()

    # Save Partition to partition.txt
    with open("partition.txt", "w") as file:
        file.write(str(partition))

    return partition

def bucketed_post_timeline(file_path, time_interval):
    with open(file_path, 'r') as file:
        json_data = file.readlines()

    data = []
    for line in json_data:
        json_object = json.loads(line)
        author = json_object.get('author')
        created_utc = int(json_object.get('created_utc'))

        if (
                author != "[deleted]"
                and author != "AutoModerator"
                and created_utc > 1546300801
                and created_utc < 1677628801
            ):
            data.append({'author': author, 'created_utc': created_utc})

    print("Bucket Subprocess - Data Loaded")

    # Create a DataFrame full of zeroes with each unique user as a row and each time period as a column, starting from 1546300801 (January 1, 2019) and ending at 1677628801 (March 1, 2023) with intervals of time_interval seconds
    df = pd.DataFrame(0, index=np.unique([row['author'] for row in data]), columns=range((1546300801 // time_interval * time_interval), (((1677628801 // time_interval) + 1) * time_interval), time_interval))


    #Fill the DataFrame with zeroes
    df = df.fillna(0)
    
    # Ensure that the author columns are unique
    assert df.index.is_unique


    progress_bar = tqdm.tqdm(total=len(data))
    # Iterate over each row in data and increment the corresponding cell in df
    for row in data:
        # Sort the "bucket" that the created_utc falls into
        bucket = (row['created_utc'] // time_interval) * time_interval
        # print(f"Author: {row['author']}, Created UTC: {row['created_utc']}, Bucket: {bucket}")
        df.at[row['author'], bucket] += 1
        progress_bar.update(1)

    progress_bar.close()

    return df

def plot_numile_contributions(file_path, time_interval, num_numiles, chosen_numile=0):
    df = bucketed_post_timeline(file_path, time_interval)
    print("1 - Bucketed")

    df_for_numiles = get_user_activity(file_path)
    print("Get User Activity Subprocess - Data Loaded")

    numile_users_dict = partition_users(df_for_numiles, calc_numiles(df_for_numiles, num_numiles))
    numile_users_list = numile_users_dict[chosen_numile]
    print("2 - Partitioned")

    # Filter DataFrame to only include users in the chosen numile
    df_in_numile = df[df.index.isin(numile_users_list)]
    print("3 - Filtered")

    # print(f"Users in numile {chosen_numile}: {df_in_numile}")

    # A dictionary of the form {utc_timestamp: [minimum, 25th percentile, median, 75th percentile, maximum]} for that time period only
    data = {}

    # Iterate over each time period
    for column in df_in_numile:
        # Get the minimum, 25th percentile, median, 75th percentile, and maximum number of posts in that time period
        data[column] = df_in_numile[column].describe()[[0, 1, 2, 3, 4, 5, 6, 7]].tolist()
    print("4 - Described")

    # Convert data to a DataFrame, with each row representing a time period and each column representing a statistic
    data_to_plot_df = pd.DataFrame.from_dict(data, orient='index', columns=['count', 'mean', 'std_dev','min', '25%', '50%', '75%', 'max'])
    print("5 - Converted Described Data to DataFrame")

    print(data_to_plot_df)

    


if __name__ == "__main__":

    num_numiles = 20

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            # print(bucketed_post_timeline(file_path, 2592000))

            # print(partition_users(get_user_activity(file_path), calc_numiles(get_user_activity(file_path), num_numiles))[5])

            # print(bucketed_post_timeline(file_path, 2592000))

            #plot_numile_contributions(file_path, 2592000, num_numiles, 8)
            plot_numile_contributions(file_path, 10520000, num_numiles, 15)
            
            # data_frame = bucketed_post_timeline(file_path, 2592000)
            # for column in data_frame:
            #     print(data_frame[column])
