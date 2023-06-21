import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from collections import Counter
import datetime

# Some Important Notes on the Code:
# 1. The code assumes that the JSON files are in the data directory
# 2. The code will only calculate numiles for users wth at least 1 submission
# 3. The code will only calculate numiles for users who have submitted after January 1, 2019
# 4. The code will calculate numiles inclusive of both upper and lower boundaries in terms of activity count

# Some Notes to Self



# Path to the directory containing JSON files
directory = "data"

def get_user_activity(file_path):

    # Initialize dictionary to store author submissions
    author_submissions = {}

    # Read JSON file
    with open(file_path) as file:
        entries = file.readlines()

    # Initialize progress bar based on JSON lines
    progress_bar = tqdm.tqdm(total=len(entries), colour="BLUE")

    # Give progress bar a description
    progress_bar.set_description("Get User Activity Subprocess - Loading Data")

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

            # Update progress bar
            progress_bar.update(1)

        except json.JSONDecodeError:
            print("Error decoding JSON entry:", entry)
            continue

    # Close progress bar
    progress_bar.close()

    # Convert author submissions dictionary to a DataFrame
    df = pd.DataFrame(list(author_submissions.items()), columns=["Author", "Submissions"])

    # Success message
    print("Get User Activity Subprocess - Data Loaded")

    return df

    # Initialize Counter object to store author submissions
    author_submissions = Counter()

    # Initiate progress bar based on JSON lines
    with open(file_path) as file:
        lines = sum(1 for line in file)
    progress_bar = tqdm.tqdm(total=lines)

    # Give progress bar a description
    progress_bar.set_description("Get User Activity Subprocess (Optimised) - Loading Data")

    # Process each entry in the JSON file
    with open(file_path) as file:
        for line in file:
            try:
                data = json.loads(line)

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
                    author_submissions[author] += 1

                # Update progress bar
                progress_bar.update(1)

            except json.JSONDecodeError:
                print("Error decoding JSON entry:", line)
                continue

    # Close progress bar
    progress_bar.close()

    # Convert author submissions Counter object to a DataFrame
    df = pd.DataFrame.from_dict(author_submissions, orient='index', columns=["Submissions"])
    df.index.name = "Author"

    # Success message
    print("Get User Activity Subprocess - Data Loaded")

    return df

def calc_numiles(df, num_numiles):
    # Numiles, returns a list of numile boundaries

    # Array of submission counts, sorted
    activity_counts = df["Submissions"].value_counts().sort_index()

    # Use pandas to calculate the num_numiles numiles
    # deciles = activity_counts.quantile(np.linspace(0, 1, num_numiles + 1))
    deciles = np.percentile(activity_counts, range(0, 101, 100//num_numiles))

    # Success message
    print("Calc Numiles Subprocess - Numiles Calculated")
    
    return deciles.tolist()

def get_numiles(file_path, num_numiles):
    df = get_user_activity(file_path)
    numiles = calc_numiles(df, num_numiles)

    # Success message
    print("Get Numiles Subprocess - Numiles Calculated")

    return numiles

def partition_users(df, numiles):

    """
    Partition users into numiles based on their activity count
    The code will update partition_info.txt with the number of users in each numile and their boundaries, 
    (NOT ENABLED DUE TO OPTIMISATION) and partition.txt with the partition dictionary
    """

    # Returns a dictionary of lists of the form {numile: [user1, user2, ...]}
    partition = {}
    for i in range(len(numiles) - 1):
        partition[i] = df[(df["Submissions"] >= numiles[i]) & (df["Submissions"] <= numiles[i + 1])]["Author"].tolist()

    # # Save Partition to partition.txt
    # with open("output/partition.txt", "w") as file:
    #     file.write(str(partition))

    # Save the number of users in each numile and their boundaries to partition_info.txt
    if not os.path.exists("output"):
        os.makedirs("output")

    with open("output/partition_info.txt", "w") as file:
        file.write("Numile\tLower Boundary\tUpper Boundary\tNumber of Users\n")
        for i in range(len(numiles) - 1):
            file.write(
                str(i)
                + "\t"
                + str(numiles[i])
                + "\t"
                + str(numiles[i + 1])
                + "\t"
                + str(len(partition[i]))
                + "\n"
            )

    # Success message
    print("Partition Users Subprocess - Users Partitioned")

    return partition

def bucketed_post_timeline(file_path, time_interval):
    with open(file_path, 'r') as file:
        json_data = file.readlines()

    # Initialise progress bar
    progress_bar = tqdm.tqdm(total=len(json_data), colour='CYAN')

    # Add descriptive text to progress bar
    progress_bar.set_description("Bucket Subprocess - Loading Data")

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
        
        progress_bar.update(1)

    # Close progress bar
    progress_bar.close()

    print("Bucket Subprocess - Data Loaded")

    # Create a DataFrame full of zeroes with each unique user as a row and each time period as a column, starting from 1546300801 (January 1, 2019) and ending at 1677628801 (March 1, 2023) with intervals of time_interval seconds
    df = pd.DataFrame(0, index=np.unique([row['author'] for row in data]), columns=range((1546300801 // time_interval * time_interval), (((1677628801 // time_interval) + 1) * time_interval), time_interval))


    #Fill the DataFrame with zeroes
    df = df.fillna(0)
    
    # Ensure that the author columns are unique
    assert df.index.is_unique

    # Initiate progress bar
    progress_bar = tqdm.tqdm(total=len(data), colour='GREEN')

    # Add a description to the progress bar
    progress_bar.set_description("Bucket Subprocess - Bucketing Data")

    # Iterate over each row in data and increment the corresponding cell in df
    for row in data:
        # Sort the "bucket" that the created_utc falls into
        bucket = (row['created_utc'] // time_interval) * time_interval
        # print(f"Author: {row['author']}, Created UTC: {row['created_utc']}, Bucket: {bucket}")
        df.at[row['author'], bucket] += 1
        progress_bar.update(1)

    progress_bar.close()

    # Success message
    print("Bucket Subprocess - Data Bucketed")

    return df

def plot_numile_contributions(file_path, time_interval, num_numiles, chosen_numile_range_lower, chosen_numile_range_upper):

    """
    params:
        file_path: path to the file containing the data
        time_interval: the time interval in seconds to bucket the data by
        num_numiles: the number of numiles to partition the data into
        chosen_numile_range_lower: the lower bound of the numile range to plot
        chosen_numile_range_upper: the upper bound of the numile range to plot

    !! NOTE: The bounds are inclusive !!

    The code will update output/partition_info.txt with the number of users in each numile and their boundaries, 
    (NOT ENABLED DUE TO OPTIMISATION) and output/partition.txt with the partition dictionary
    The code will also update output/plot_summary_stat.tsv with the summary statistics for the particular numile chosen
    """

    df_bucketed_posts = bucketed_post_timeline(file_path, time_interval)

    df_user_activity = get_user_activity(file_path)

    numile_users_dict = partition_users(df_user_activity, calc_numiles(df_user_activity, num_numiles))

    # Get all users in the chosen numile range
    numile_users_list = []
    for i in range(chosen_numile_range_lower, chosen_numile_range_upper + 1):
        numile_users_list.extend(numile_users_dict[i])
    

    # Filter DataFrame to only include users in the chosen numile
    df_in_numile = df_bucketed_posts[df_bucketed_posts.index.isin(numile_users_list)]
    print("Plot Numile Contributions Subprocess - Data Filtered")


    # A dictionary of the form {utc_timestamp: [minimum, 25th percentile, median, 75th percentile, maximum]} for that time period only
    data = {}

    # Iterate over each time period
    for column in df_in_numile:
        # Get the minimum, 25th percentile, median, 75th percentile, and maximum number of posts in that time period
        data[column] = df_in_numile[column].describe()[[0, 1, 2, 3, 4, 5, 6, 7]].tolist()
    print("Plot Numile Contributions Subprocess - Data Described")

    # Convert data to a DataFrame, with each row representing a time period and each column representing a statistic
    data_to_plot_df = pd.DataFrame.from_dict(data, orient='index', columns=['count', 'mean', 'std_dev','min', '25%', '50%', '75%', 'max'])
    print("Plot Numile Contributions Subprocess - Data Converted to DataFrame")

    # Save data_to_plot_df to a TSV file with the summary statistics for each time period
    data_to_plot_df.to_csv(f"output/plot_summary_stat.tsv", sep="\t")


    ##### ------ PLOT ------ #####

    # Convert the Unix timestamp to datetime format
    data_to_plot_df['time'] = pd.to_datetime(data_to_plot_df.index, unit='s')

    # Set the 'time' column as the index
    data_to_plot_df.set_index('time', inplace=True)

    # Plotting the box plot
    plt.figure(figsize=(10, 6))
    data_to_plot_df[['min', '25%', '50%', '75%', 'max']].plot(kind='box')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Box Plot of Min, 25%, 50%, 75%, Max')
    plt.xticks(rotation=45)

    if not os.path.exists("figures"):
        os.makedirs("figures")

    plt.savefig(f"figures/box_plot_numile_{chosen_numile_range_lower}_{chosen_numile_range_upper}_{time_interval}.png")

    





# Main function
if __name__ == "__main__":

    num_numiles = 20

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            # print(bucketed_post_timeline(file_path, 2592000))

            # user_activity_df = get_user_activity(file_path)
            # partition_users(user_activity_df, calc_numiles(user_activity_df, num_numiles))[5]
            
            # print(bucketed_post_timeline(file_path, 2592000))

            #plot_numile_contributions(file_path, 2592000, num_numiles, 8)
            plot_numile_contributions(file_path, 10520000, num_numiles, 19, 19)
            
            # data_frame = bucketed_post_timeline(file_path, 2592000)
            # for column in data_frame:
            #     print(data_frame[column])
