import pandas as pd
import json
import os

# List of CSV files to process
csv_files = [
    '../data/processed_data/processed_csv/depression_dataset_reddit_cleaned_processed.csv',
    '../data/processed_data/processed_csv/dreaddit_merged_processed.csv',
    '../data/processed_data/processed_csv/Mental_Health_FAQ_processed.csv',
    '../data/raw_data/CSV Files/context_response_train.csv'
]

# Directory to save JSON files
output_json_dir = '../data/processed_data/processed_JSON/'

# Create the output directory if it doesn't exist
os.makedirs(output_json_dir, exist_ok=True)

# Function to convert CSV to JSON format
def csv_to_json_individual(csv_files, output_json_dir):
    # Loop through the CSV files
    for csv_file in csv_files:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Convert the DataFrame to a list of dictionaries (records)
        data = df.to_dict(orient='records')
        
        # Remove square brackets from all string values in the records
        for record in data:
            for key, value in record.items():
                if isinstance(value, str):
                    record[key] = value.replace('[', '').replace(']', '')
        
        # Generate a JSON file name based on the CSV file name
        file_name = os.path.basename(csv_file).replace('.csv', '.json')
        output_json_path = os.path.join(output_json_dir, file_name)
        
        # Save the data to a JSON file
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        
        print(f"CSV data successfully converted and saved to {output_json_path}")

# Call the function to process the CSV files
csv_to_json_individual(csv_files, output_json_dir)
