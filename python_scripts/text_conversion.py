import os
import json

def json_to_text(json_file_path, text_file_path):
    # Read the JSON file
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        
    # Write the JSON data to a text file
    with open(text_file_path, mode='w', encoding='utf-8') as text_file:
        # Convert the JSON data to a readable format and write it
        json.dump(data, text_file, ensure_ascii=False, indent=4)
    
    print(f"JSON data successfully converted and saved to {text_file_path}")

def convert_all_json_to_text(json_dirs, text_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    
    # Loop over all specified JSON directories
    for json_dir in json_dirs:
        # Loop through all files in the JSON directory
        for json_file_name in os.listdir(json_dir):
            if json_file_name.endswith('.json'):
                json_file_path = os.path.join(json_dir, json_file_name)
                text_file_name = json_file_name.replace('.json', '.txt')
                text_file_path = os.path.join(text_dir, text_file_name)
                
                # Convert each JSON file to a text file
                json_to_text(json_file_path, text_file_path)
    
    print("All JSON files converted to text.")

# Example usage
json_dirs = ["../data/raw_data/JSON Files", "../data/processed_data/processed_JSON"]  # List of JSON directories
text_dir = "../data/processed_data/processed_text"  # Directory to save the text files

convert_all_json_to_text(json_dirs, text_dir)
