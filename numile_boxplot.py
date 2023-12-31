import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import tqdm
from collections import Counter
import datetime
import time

# Some Important Notes on the Code:
# 1. The code assumes that the JSON files are in the data directory
# 2. The code will only calculate numiles for users wth at least 1 submission
# 3. The code will only calculate numiles for users who have submitted after January 1, 2019
# 4. The code will calculate numiles inclusive of both upper and lower boundaries in terms of activity count

# Some Notes to Self
# 1. Pallete of colours for progress bar: https://coolors.co/e4fde1-8acb88-648381-575761-ffbf46



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


def bucketed_post_timeline(file_path, time_interval, end_time=1677628801):
    with open(file_path, 'r') as file:
        json_data = file.readlines()

    # Initialise progress bar
    progress_bar = tqdm.tqdm(total=len(json_data), colour='#648381')

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

    # Create a DataFrame full of zeroes with each unique user as a row and each time period as a column, 
    # starting from 1546300801 (January 1, 2019) and ending at 1677628801 (March 1, 2023) by defualt (or end_time param) 
    # with intervals of time_interval seconds
    df = pd.DataFrame(0, index=np.unique([row['author'] for row in data]), columns=range((1546300801 // time_interval * time_interval), (((end_time // time_interval) + 1) * time_interval), time_interval))


    #Fill the DataFrame with zeroes
    df = df.fillna(0)
    
    # Ensure that the author columns are unique
    assert df.index.is_unique

    # Initiate progress bar
    progress_bar = tqdm.tqdm(total=len(data), colour='#8acb88')

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

    start = time.time()

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


    ##### ------ SUMMARY STATISTICS ------ #####

    # A dictionary of the form {utc_timestamp: [minimum, 25th percentile, median, 75th percentile, maximum]} for that time period only
    summary_data = {}

    # Iterate over each time period
    for column in df_in_numile:
        # Get the minimum, 25th percentile, median, 75th percentile, and maximum number of posts in that time period
        summary_data[column] = df_in_numile[column].describe()[[0, 1, 2, 3, 4, 5, 6, 7]].tolist()
    print("Plot Numile Contributions Subprocess - Data Described")

    # Convert data to a DataFrame, with each row representing a time period and each column representing a statistic
    data_summary_statistics = pd.DataFrame.from_dict(summary_data, orient='index', columns=['count', 'mean', 'std_dev','min', '25%', '50%', '75%', 'max'])
    print("Plot Numile Contributions Subprocess - Data Converted to DataFrame")

    # Save data_to_plot_df to a TSV file with the summary statistics for each time period
    data_summary_statistics.to_csv(f"output/plot_summary_stat.tsv", sep="\t")


    ##### ------ PLOT DATA GEN ------ #####

    # A dictionary of the form {utc_timestamp: [datapoint1, datapoint2, ...]} for that time period only
    plot_data = {}

    # Iterate over each time period
    for column in df_in_numile:
        # Get the number of posts in that time period
        plot_data[column] = df_in_numile[column].tolist()
    print("Plot Numile Contributions Subprocess - Data Converted to Dictionary")

    # Save plot_data to a TSV file
    with open(f"output/plot_data.tsv", 'w') as f:
        for key in plot_data.keys():
            f.write(f"{key}\t{plot_data[key]}\n")


    ##### ------ PLOT ------ #####

    # Create a new figure with a single subplot
    fig, ax = plt.subplots()

    # Extract the utc_timestamps and datapoints from the dictionary
    utc_timestamps = list(plot_data.keys())
    datapoints = list(plot_data.values())

    # Omit Zeroes
    datapoints = [[x for x in y if x != 0] for y in datapoints]

    # Plot the box plot
    ax.boxplot(datapoints, showfliers=False)

    # Set the x-axis labels to the utc_timestamps in year-month-day format
    ax.set_xticks(range(1, len(utc_timestamps) + 1))
    ax.set_xticklabels([datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d') for x in utc_timestamps], rotation=90)

    # Set labels and title
    ax.set_xlabel('Interval Start Date')
    ax.set_ylabel('Submissions/Comments per User')
    ax.set_title(f'Box Plot of Userly Submissions/Comments \nOver UTC Timestamps for Numiles {chosen_numile_range_lower} to {chosen_numile_range_upper} out of {num_numiles} Numiles')

    # Add subtitle with the numile interval, time interval, and count of users in the chosen numile range
    ax.text(0.5, 0.90, f"File Path: {file_path}\nNumile Interval (inclusive): {chosen_numile_range_lower} to {chosen_numile_range_upper} out of {num_numiles} Numiles\nTime Interval: {time_interval / 2592000} month(s)\nNumber of Users: {len(numile_users_list)}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Add margin to the figure
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9)

    # Set the plot size
    fig.set_size_inches(15, 7)

    # Create the figures directory if it does not exist
    if not os.path.exists("figures"):
        os.makedirs("figures")

    fig.savefig(f"figures/box_plot_numile_{chosen_numile_range_lower}_{chosen_numile_range_upper}_{time_interval}.png")

    end = time.time()
    print(f"Plot Total Time Taken: {end - start} seconds")


def plot_bounded(file_path, time_interval, lower_bound, upper_bound, start_date=1546300801, end_date=1677628801):

    start = time.time()

    # Print the file name being processed
    print(f"Starting on {os.path.basename(file_path[:-5])}")

    df_bucketed_posts = bucketed_post_timeline(file_path, time_interval)

    # Initiate a progress bar
    pbar = tqdm.tqdm(total=len(df_bucketed_posts.index), colour='#E4FDE1', desc='Plot Bounded Subprocess - Calculating Total Posts per User')

    # Get a dictionary of total number of posts per user within the start and end date in the form {user: total_posts}
    user_posts_dict = {}
    for i in range(len(df_bucketed_posts.index)):
        user_posts_dict[df_bucketed_posts.index[i]] = df_bucketed_posts.iloc[i].sum()
        pbar.update(1)
    pbar.close()

    # Get the list of users between the bounds
    bounded_user_list = [user for user in user_posts_dict.keys() if lower_bound <= user_posts_dict[user] <= upper_bound]

    # Filter DataFrame to only include users in the chosen numile
    df_in_bounds = df_bucketed_posts[df_bucketed_posts.index.isin(bounded_user_list)]
    print("Plot Bounded Subprocess - Data Filtered")

    ##### ------ SUMMARY STATISTICS ------ #####

    # A dictionary of the form {utc_timestamp: [minimum, 25th percentile, median, 75th percentile, maximum]} for that time period only
    summary_data = {}

    # Iterate over each time period
    for column in df_in_bounds:
        # Get the minimum, 25th percentile, median, 75th percentile, and maximum number of posts in that time period
        summary_data[column] = df_in_bounds[column].describe()[[0, 1, 2, 3, 4, 5, 6, 7]].tolist()
    print("Plot Bounded Subprocess - Data Described")

    # Convert data to a DataFrame, with each row representing a time period and each column representing a statistic
    data_summary_statistics = pd.DataFrame.from_dict(summary_data, orient='index', columns=['count', 'mean', 'std_dev','min', '25%', '50%', '75%', 'max'])
    print("Plot Bounded Subprocess - Data Converted to DataFrame")

    # Save data_to_plot_df to a TSV file with the summary statistics for each time period
    data_summary_statistics.to_csv(f"output/plot_bounded_summary_stat.tsv", sep="\t")


    ##### ------ PLOT DATA GEN ------ #####

    # A dictionary of the form {utc_timestamp: [datapoint1, datapoint2, ...]} for that time period only
    plot_data = {}

    # Iterate over each time period
    for column in df_in_bounds:
        # Get the number of posts in that time period
        plot_data[column] = df_in_bounds[column].tolist()
    print("Plot Bounded Subprocess - Data Converted to Dictionary")

    # Save plot_data to a TSV file
    with open(f"output/plot_data.tsv", 'w') as f:
        for key in plot_data.keys():
            f.write(f"{key}\t{plot_data[key]}\n")


    ##### ------ PLOT ------ #####

    # Create a new figure with a single subplot
    fig, ax = plt.subplots()

    # Extract the utc_timestamps and datapoints from the dictionary
    utc_timestamps = list(plot_data.keys())
    datapoints = list(plot_data.values())

    # Omit Zeroes
    datapoints = [[x for x in y if x != 0] for y in datapoints]

    # Plot the box plot
    ax.boxplot(datapoints, showfliers=False)

    # Set the x-axis labels to the utc_timestamps in year-month-day format
    ax.set_xticks(range(1, len(utc_timestamps) + 1))
    ax.set_xticklabels([datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d') for x in utc_timestamps], rotation=90)

    # Set labels and title
    ax.set_xlabel('Interval Start Date')
    ax.set_ylabel('Submissions/Comments per User')
    ax.set_title(f'Box Plot of Userly Submissions/Comments \nOver UTC Timestamps for Bounds {lower_bound} to {upper_bound} \nFrom {pd.to_datetime(start_date, unit="s").strftime("%Y-%m-%d")} to {pd.to_datetime(end_date, unit="s").strftime("%Y-%m-%d")}')

    # Add subtitle with the bound interval, time interval integer in human readable format, count of users in the chosen bound range, and the lower and upper time integers in UTC timestamp bounds in human readable format using pandas to_datetime
    ax.text(0.8, 1.15, f"Bound Interval: {lower_bound} to {upper_bound} \nTime Interval: {time_interval / 2592000} month(s) \nNumber of Users in Bound: {len(bounded_user_list)} \nStart Date: {pd.to_datetime(start_date, unit='s').strftime('%Y-%m-%d')} \nEnd Date: {pd.to_datetime(end_date, unit='s').strftime('%Y-%m-%d')}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Add margin to the figure
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.8)

    # Set the plot size
    fig.set_size_inches(15, 9)

    # Create the figures directory if it does not exist
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Save the figure with the name of the processed JSON file and lower upper bounds
    fig.savefig(f"figures/box_plot_bounds_{lower_bound}_{upper_bound}_{os.path.basename(file_path[:-5])}.png", dpi=300)

    #fig.savefig(f"figures/box_plot_bounds_{lower_bound}_{upper_bound}.png", dpi=300)

    end = time.time()
    print(f"Plot Bounded Total Time Taken: {end - start} seconds")
    



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

            # plot_numile_contributions(file_path, 10520000, num_numiles, 17, 18)
            
            # data_frame = bucketed_post_timeline(file_path, 2592000)
            # for column in data_frame:
            #     print(data_frame[column])

            for bound_interval in [[1, 10],[11, 100], [101, 1000], [1000, 1000000000]]:
                plot_bounded(file_path, 10520000, bound_interval[0], bound_interval[1], 1546300801, 1677628801)

    # for numile_interval in [[19, 19], [17, 18], [15, 16], [13, 14], [11, 12], [9, 10], [7, 8], [5, 6], [3, 4], [1, 2]]:
    #     plot_numile_contributions(file_path, 10520000, num_numiles, numile_interval[0], numile_interval[1])

    
