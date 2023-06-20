import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to the directory containing JSON files
directory = "data"
numiles = 20

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)

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
        print(df.sort_values(by="Submissions", ascending=False))

        # Plotting
        activity_counts = df["Submissions"].value_counts().sort_index()
        print(f"Activity counts for {filename}: {activity_counts}")
        print(activity_counts)
        plt.figure(figsize=(18,6))
        plt.bar(activity_counts.index, activity_counts.values, width=0.8)
        plt.xlabel("User Activity")
        plt.ylabel("Number of Users")
        plt.yscale("log")
        plt.title(f"User Activity vs. Number of Users - {filename}")
        plt.xticks(rotation=45, ha="right")
        plt.locator_params(axis="x")

        # Calculate deciles
        deciles = np.percentile(activity_counts.values, range(0, 100, 100//numiles))

        # Add vertical lines for deciles
        for decile in deciles:
            decile_label = "{:.2f}".format(decile)
            plt.axvline(decile, color='r', linestyle='--')
            plt.text(
                decile,
                0.7 * plt.ylim()[1],
                decile_label,
                rotation=90,
                va='top',
                ha='center',
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5)
            )

        print(f"Numiles ({numiles}):", deciles)

        plt.tight_layout()
        plt.savefig(f"{filename}.png")