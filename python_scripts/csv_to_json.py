import csv
import json
import os

def csv_to_json(csv_file_path, json_file_path):
    data = []
    
    # Read the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Loop over each row and append it to the data list
        for row in csv_reader:
            data.append({
                "context": row["Context"],
                "response": row["Response"]
            })
    
    # Write the data to a JSON file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    
    print(f"CSV data successfully converted and saved to {json_file_path}")

# Example usage
csv_file_path = "../data/raw_data/CSV Files/context_response_train.csv"  # Replace with your CSV file path
json_file_path = "../data/processed_data/processed_json/context_response_train.json"  # Replace with your desired output JSON file path
csv_to_json(csv_file_path, json_file_path)

def csv_to_json2(csv_file_path, json_file_path, key_col, value_col):
    data = []
    
    # Read the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Loop over each row and append key-value pairs to the data list
        for row in csv_reader:
            data.append({
                row[key_col]: row[value_col]
            })
    
    # Write the data to a JSON file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    
    print(f"CSV data from {csv_file_path} successfully converted and saved to {json_file_path}")

def process_all_csv_files(csv_dir, json_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    # Loop through all files in the CSV directory
    for csv_file_name in os.listdir(csv_dir):
        csv_file_path = os.path.join(csv_dir, csv_file_name)
        json_file_path = os.path.join(json_dir, csv_file_name.replace('.csv', '.json'))
        
        # Check which file is being processed and set appropriate key-value mapping
        if csv_file_name == 'depression_dataset_reddit_cleaned_processed.csv':
            csv_to_json2(csv_file_path, json_file_path, key_col='is_depression', value_col='clean_text')
        elif csv_file_name == 'dreaddit_merged_processed.csv':
            csv_to_json2(csv_file_path, json_file_path, key_col='subreddit', value_col='text')
        elif csv_file_name == 'Mental_Health_FAQ_processed.csv':
            csv_to_json2(csv_file_path, json_file_path, key_col='Questions', value_col='Answers')
        else:
            print(f"Skipping file: {csv_file_name} - No key-value mapping specified.")
    
    print("All CSV files processed.")

# Example usage
csv_dir = "../data/processed_data/processed_csv"  # Directory containing CSV files
json_dir = "../data/processed_data/processed_JSON"  # Directory to save JSON files

process_all_csv_files(csv_dir, json_dir)
