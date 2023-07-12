# Set the Reddit Data File Path (JSON)
data_file_path = "data/texas_comments.json"

time_interval_seconds = 2 * 30 * 24 * 60 * 60 # two months
time_start = 1546300801 # 1st January 2019
time_end = 1672531201 # 1st January 2023

activity_bucket_limits = [1, 10, 100, 1000, 10000, 100000, 1000000] # Inclusive on lower bound, exclusive on upper bound - strictly integers



import liwc
parse, category_names = liwc.load_token_parser('dictionaries/LIWC07-EN.dic')




import re
from collections import Counter

def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

def count_categories(text):
    # Returns a Counter object containing tallies of all LIWC categories
    text = text.lower()

    # Remove all words in text that are less than 3 characters long
    text = re.sub(r'\b\w{1,3}\b', '', text)

    # Remove "[removed]" and "[deleted]" from the body text
    text = re.sub(r'\[removed\]|\[deleted\]', '', text)
    
    tokens = tokenize(text)
    counter = Counter(category for token in tokens for category in parse(token))

    # If a category doesn't exist in the text, we need to add it and set the count to 0
    for category in category_names:
        if category not in counter:
            counter[category] = 0



    return counter

gettysburg = "hello world, this [removed]"

print(count_categories(gettysburg))


import tqdm
import pandas as pd
import json
import os


# Group the users into buckets based on activity during each time interval
# Create a list of dictionaries the following format: (the index in the list is the time bucket index)
# [{user_id: posts_during_time_interval, user_id: posts_during_time_interval, ...}, 
#  {user_id: posts_during_time_interval, user_id: posts_during_time_interval, ...}, ...]

# Initialise a list of empty dictionaries of the amount of time intervals
activity_buckets = [{} for i in range(int((time_end - time_start) / time_interval_seconds) + 1)]

# Initialise progress bar
pbar = tqdm.tqdm(total=os.path.getsize(data_file_path))

# Iterate through the data file
with open(data_file_path) as data_file:
    # Analyse each line in the data file
    for line in data_file:
        # Load the line as a JSON object
        line_json = json.loads(line)
        
        # Get the timestamp of the comment
        timestamp = int(line_json["created_utc"])
        user_id = line_json["author"]

        # Check if within start and end time
        if timestamp < time_start or timestamp > time_end:
            # Update the progress bar
            pbar.update(len(line))
            continue
        
        # Get the time interval index
        time_interval_index = int((timestamp - time_start) / time_interval_seconds)

        # Check if the user has already been added to the activity bucket
        if user_id in activity_buckets[time_interval_index]:
            # Increment the user's post count
            activity_buckets[time_interval_index][user_id] += 1
        else:
            # Add the user to the activity bucket
            activity_buckets[time_interval_index][user_id] = 1

        # Update the progress bar
        pbar.update(len(line))

# Close the progress bar
pbar.close()

# Initialise another progress bar
pbar = tqdm.tqdm(total=len(activity_buckets))

# Change the activity count into bucket indices
for time_bucket in activity_buckets:
    # Update the progress bar
    pbar.update(1)
    for user in time_bucket:
        # Iterate through the activity bucket limits
        for i in range(len(activity_bucket_limits)):
            # Using the activity bucket limits, find the bucket index
            if time_bucket[user] >= activity_bucket_limits[i] and time_bucket[user] < activity_bucket_limits[i + 1]:
                # Set the activity bucket to the bucket index
                time_bucket[user] = i
                break


# Close the progress bar
pbar.close()

activity_buckets




run_CSV_creation = True

if run_CSV_creation:
    # Intialise the progress bar based on the number of items in the data file
    with open(data_file_path) as f:
        num_lines = sum(1 for line in f)

    pbar = tqdm.tqdm(total=num_lines)

    # Read the data file and add the data to the csv file
    with open(data_file_path) as f:
        with open(f'text_liwc_dimensions/{os.path.basename(data_file_path[:-5])}_LIWC.csv', 'w') as csv_file:

            # Clear the csv file of any existing data
            csv_file.truncate(0)

            for line in f:
                LIWC_by_bucket = json.loads(line)
                utc_timestamp = LIWC_by_bucket['created_utc']
                body_text = LIWC_by_bucket['body']
                LIWC_categories = count_categories(body_text)
                author = LIWC_by_bucket['author']

                # Check if post is within the time interval
                if int(utc_timestamp) < time_start or int(utc_timestamp) > time_end:
                    pbar.update(1)
                    continue

                # Determine the activity bucket index
                time_interval_index = int((int(utc_timestamp) - time_start) / time_interval_seconds)
                activity_buckets_index = activity_buckets[time_interval_index][author]

                # Cleanse the body_text of any newlines, commas, and quotation marks, and replace them with spaces
                body_text = body_text.replace('\n', ' ').replace(',', ' ').replace('"', ' ')


                # Write the data to the csv file
                csv_file.write(f'{utc_timestamp},"{LIWC_categories}", {activity_buckets_index}, "{body_text}"\n')

                # Update the progress bar
                pbar.update(1)
            

    # Close the progress bar
    pbar.close()







# Load the data from the csv file into a dataframe
df = pd.read_csv(f'text_liwc_dimensions/{os.path.basename(data_file_path[:-5])}_LIWC.csv', lineterminator='\n')

# Add headers to the dataframe
df.columns = ['utc_timestamp', 'LIWC_categories', 'activity_bucket_index', 'body_text']

df.head()




import numpy as np

# Calculate time bucket intervals
time_buckets = range(int(time_start), int(time_end) + int(time_interval_seconds), int(time_interval_seconds))

# Print maximum utc_timestamp
# print(df['utc_timestamp'].max())
# print("------------------")
# print(*time_buckets, sep='\n')


# Change the utc_timestamp to the bucket number, e.g. the 0th bucket is between the 0th index and the 1st index of the time_buckets
df['utc_timestamp'] = df['utc_timestamp'].apply(lambda x: np.searchsorted(time_buckets, x) - 1)


df.head()






# Graph the number of comments per time bucket
import matplotlib.pyplot as plt

# Delete all rows with a utc_timestamp of -1
df = df[df['utc_timestamp'] != -1]

df['utc_timestamp'].value_counts().sort_index().plot(kind='bar', figsize=(20, 10))
plt.show()




# Graph the number of comments per time bucket
import matplotlib.pyplot as plt

# The LIWC categories stored in a list
data = [
    'funct', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they',
    'ipron', 'article', 'verb', 'auxverb', 'past', 'present', 'future',
    'adverb', 'preps', 'conj', 'negate', 'quant', 'number', 'swear',
    'social', 'family', 'friend', 'humans', 'affect', 'posemo',
    'negemo', 'anx', 'anger', 'sad', 'cogmech', 'insight', 'cause',
    'discrep', 'tentat', 'certain', 'inhib', 'incl', 'excl', 'percept',
    'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest',
    'relativ', 'motion', 'space', 'time', 'work', 'achieve', 'leisure',
    'home', 'money', 'relig', 'death', 'assent', 'nonfl', 'filler'
]



def graph_category(category):
    # Construct a dictionary with the value in the form of a nested list, and the key as the LIWC category
    # where the outer index is the activity_bucket_index, and the inner index is the time_bucket_index
    # {LIWC_category: [[time_bucket_0, time_bucket_1, ...], [time_bucket_0, time_bucket_1, ...], ...]}

    print(f"Commencing on {category}")
    # Initialise the dictionary
    LIWC_category_dict = {}

    # Initialise the dictionary with empty nested lists
    LIWC_category_dict = {cat: [[0 for _ in time_buckets] for _ in activity_bucket_limits] for cat in data}

    # print(LIWC_category_dict)

    # Start a progress bar and label it "aggregating LIWC categories for {category}"
    pbar = tqdm.tqdm(total=len(df), desc=f'aggregating LIWC categories for {category}')

    # Iterate through each entry in df and aggregate the LIWC categories for each time_bucket
    for index, row in df.iterrows():
        # Get the time_bucket and LIWC_categories
        time_bucket = row['utc_timestamp']
        LIWC_categories = row['LIWC_categories']
        activity_bucket_index = row['activity_bucket_index']

        # Convert the LIWC_categories string to a dictionary
        LIWC_categories = eval(LIWC_categories)

        # A percentage of all LIWC words must be the selected category
        # Check that LIWC_categories[selected_category] is not zero to avoid division by zero
        if LIWC_categories[category] != 0:
            if not LIWC_categories[category] / sum(LIWC_categories.values()) > 0.05:
                # Update the progress bar
                pbar.update(1)
                continue
                

        # For each row, go through the LIWC categories and add the time_bucket to the correct list
        for category in LIWC_categories:
            LIWC_category_dict[category][activity_bucket_index][time_bucket] += (LIWC_categories[category])

        # Update the progress bar
        pbar.update(1)

    # Close the progress bar
    pbar.close()




    ####---------------------------- Normalising ---------------------------- #####
    # Normalise the data by dividing each value by the total number of LIWC detected words in that activity and time bucket

    # Get the sum of all the LIWC categories for each activity and time bucket in the form of a nested list, the outer index is the activity_bucket_index, and the inner index is the time_bucket_index
    # [[time_bucket_0, time_bucket_1, ...], [time_bucket_0, time_bucket_1, ...], ...]

    print(f"Normalising on {category}")

    LIWC_category_dict_normalised = LIWC_category_dict

    LIWC_category_sums = [[0 for _ in time_buckets] for _ in activity_bucket_limits]

    for activity_bucket_index in range(len(activity_bucket_limits)):
        for time_bucket_index in range(len(time_buckets)):
            # Sum the LIWC categories for each activity and time bucket
            LIWC_category_sums[activity_bucket_index][time_bucket_index] = sum(LIWC_category_dict[category][activity_bucket_index][time_bucket_index] for category in data)

    # Normalise the data by dividing each value by the total number of LIWC detected words in that activity and time bucket
    for activity_bucket_index in range(len(activity_bucket_limits)):
        for time_bucket_index in range(len(time_buckets)):
            for category in data:
                # Normalise the data
                # If sum is zero, set to zero, otherwise divide
                if LIWC_category_sums[activity_bucket_index][time_bucket_index] == 0:
                    LIWC_category_dict_normalised[category][activity_bucket_index][time_bucket_index] = 0
                else:
                    LIWC_category_dict_normalised[category][activity_bucket_index][time_bucket_index] /= LIWC_category_sums[activity_bucket_index][time_bucket_index]


    # Save LIWC_category_dict_normalised as a literal string to a file
    with open(f'text_liwc_dimensions/{os.path.basename(data_file_path[:-5])}_LIWC_normalised.json', 'w') as f:
        f.write(str(LIWC_category_dict_normalised))



    ####---------------------------- Graphing ---------------------------- #####

    print(f"Graphing on {category}")

    # Read LIWC_category_dict_normalised as a literal string from a file
    with open(f'text_liwc_dimensions/{os.path.basename(data_file_path[:-5])}_LIWC_normalised.json', 'r') as f:
        # Read the file as a string
        LIWC_category_dict_normalised = f.read()

        # Convert the string to a dictionary
        LIWC_category_dict_normalised = eval(LIWC_category_dict_normalised)

    # Import datetime to convert the time_buckets to human readable labels
    from datetime import datetime

    # Extract the data from the dictionary
    LIWC_by_bucket = LIWC_category_dict_normalised[category]

    # Set plot size
    plt.figure(figsize=(20, 10))

    # Graph all activity buckets in different colours
    for i in range(3):
        # Create a moving average by taking the average of the current, previous and next time bucket
        moving_average = [(LIWC_by_bucket[i][j] + LIWC_by_bucket[i][j - 1] + LIWC_by_bucket[i][j + 1]) / 3 for j in range(1, len(time_buckets) - 1)]
        # Plot the moving average
        plt.plot(time_buckets[1:-2], moving_average[:-1], label=f'Activity Bucket {i}')


    # Give time buckets human readable labels
    plt.xticks(time_buckets[:-1], [datetime.utcfromtimestamp(time_bucket).strftime('%Y-%m-%d %H:%M:%S') for time_bucket in time_buckets[:-1]], rotation=90)

    print(f"Adding elements to {category}")

    # Add labels and title
    plt.xlabel('Time Bucket Start Timestamp')
    plt.ylabel('Proportion of Selected Category over all LIWC Detected Words in Time Bucket')
    plt.title(f'Proportion of {category} Over Time for Dataset {os.path.basename(data_file_path[:-5])}')

    # Add a legend
    plt.legend()

    # Display the graph
    plt.savefig(f'figures/allLIWC/{os.path.basename(data_file_path[:-5])}_LIWC_normalised_{category}.png')


for category in data:
    graph_category(category)






