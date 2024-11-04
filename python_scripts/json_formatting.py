import json
import os

# Define the directory containing the JSON files and the output directory
input_directory = '../data/raw_data/JSON Files'  # Replace with your input directory
output_directory = '../data/processed_data/processed_JSON'  # Replace with your output directory

def process_json_files(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            # Read the JSON file
            with open(input_file_path, 'r') as infile:
                data = json.load(infile)

            # Check if the JSON data is a list (list of dictionaries)
            if isinstance(data, list):
                # Write the data to the output file in the desired format
                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    json.dump(data, outfile, indent=4, ensure_ascii=False)
                
                print(f"Processed file: {filename} and saved to {output_file_path}")
            else:
                print(f"Skipping file {filename}, data is not a list of dictionaries.")

# Call the function to process all JSON files
process_json_files(input_directory, output_directory)
