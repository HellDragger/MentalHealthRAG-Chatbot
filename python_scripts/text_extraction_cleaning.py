import os
import re
import json
import pandas as pd
import PyPDF2

# Directory paths
pdf_dir = 'data/raw_data/PDF Files'
csv_dir = 'data/raw_data/CSV Files'
json_dir = 'data/raw_data/JSON Files'
output_dir = 'data/processed_data/text'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to clean text
def clean_text(text):
    # Remove headers, footers, and unnecessary elements using regex
    text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Process PDF files
def process_pdf_files(pdf_dir, output_dir):
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            output_path = os.path.join(output_dir, pdf_file.replace('.pdf', '.txt'))

            # Extract text using PyPDF2
            text = ''
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + '\n'
            
            cleaned_text = clean_text(text)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)

# Process CSV files
def process_csv_files(csv_dir, output_dir):
    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(csv_dir, csv_file)
            output_path = os.path.join(output_dir, csv_file.replace('.csv', '.txt'))

            # Load CSV using pandas
            df = pd.read_csv(csv_path)

            # Clean data: handle missing values, inconsistencies
            df.fillna('', inplace=True)
            cleaned_df = df.applymap(lambda x: str(x).strip())

            cleaned_text = cleaned_df.to_string(index=False)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)

# Process JSON files
def process_json_files(json_dir, output_dir):
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            output_path = os.path.join(output_dir, json_file.replace('.json', '.txt'))

            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Clean data: handle missing values, inconsistencies
            def clean_dict(d):
                if isinstance(d, dict):
                    return {k: clean_dict(v) for k, v in d.items() if v}
                elif isinstance(d, list):
                    return [clean_dict(i) for i in d if i]
                else:
                    return str(d).strip()

            cleaned_data = clean_dict(data)
            cleaned_text = json.dumps(cleaned_data, indent=4)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)

# Run the processes
process_pdf_files(pdf_dir, output_dir)
process_csv_files(csv_dir, output_dir)
process_json_files(json_dir, output_dir)

print("Data cleaning completed!")
