import pandas as pd
import os

# File paths
input_dir = "../data/raw_data/CSV Files"
output_dir = "../data/processed_data/processed_csv/"
os.makedirs(output_dir, exist_ok=True)

# 1. Processing depression_dataset_reddit_cleaned.csv
def process_depression_dataset(file_path, output_dir):
    df = pd.read_csv(file_path)
    # Map 1 to 'depression' and 0 to 'not depression'
    df['is_depression'] = df['is_depression'].map({1: 'depression', 0: 'not depression'})
    df.to_csv(os.path.join(output_dir, 'depression_dataset_reddit_cleaned_processed.csv'), index=False)
    print("Processed depression_dataset_reddit_cleaned_processed.csv")


# 2. Merging and processing dreaddit-test.csv and dreaddit-train.csv
def process_dreaddit(file1, file2, output_dir):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Concatenate the two dataframes
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # Modify the 'subreddit' column where 'label' is 0
    merged_df['subreddit'] = merged_df.apply(lambda row: f"not {row['subreddit']}" if row['label'] == 0 else row['subreddit'], axis=1)
    
    # Select only the 'subreddit' and 'text' columns
    merged_df = merged_df[['subreddit', 'text']]
    
    # Save the merged and modified dataframe to a CSV file
    merged_df.to_csv(os.path.join(output_dir, 'dreaddit_merged_processed.csv'), index=False)
    print("Processed dreaddit-test and dreaddit-train merged")


# 3. Processing Mental_Health_FAQ.csv
def process_mental_health_faq(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Question_ID'])
    df.to_csv(os.path.join(output_dir, 'Mental_Health_FAQ_processed.csv'), index=False)
    print("Processed Mental_Health_FAQ.csv")

# File processing
process_depression_dataset(os.path.join(input_dir, 'depression_dataset_reddit_cleaned.csv'),output_dir)
process_dreaddit(os.path.join(input_dir, 'dreaddit-test.csv'), os.path.join(input_dir, 'dreaddit-train.csv'), output_dir)
process_mental_health_faq(os.path.join(input_dir, 'Mental_Health_FAQ.csv'))