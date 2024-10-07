import os
import json
from PyPDF2 import PdfReader

# Function to extract text from PDF files and save as JSON
def pdf_folder_to_json(pdf_folder_path, output_json_path):
    pdf_data = {}

    # Loop through all files in the folder
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            pdf_file_path = os.path.join(pdf_folder_path, filename)
            
            # Read the PDF file
            reader = PdfReader(pdf_file_path)
            text = ""
            
            # Extract text from each page
            for page in reader.pages:
                text += page.extract_text()
            
            # Add the extracted text to the dictionary with the file name as the key
            pdf_data[filename] = text.strip()

    # Save the dictionary as a JSON file
    with open(output_json_path, mode='w', encoding='utf-8') as json_file:
        json.dump(pdf_data, json_file, ensure_ascii=False, indent=4)
    
    print(f"PDF data successfully converted and saved to {output_json_path}")

# Example usage
pdf_folder_path = "../data/raw_data/PDF_Files"  # Replace with your folder path containing PDF files
output_json_path = "../data/processed_data/processed_json/pdfs.json"   # Replace with the path to save the JSON file
pdf_folder_to_json(pdf_folder_path, output_json_path)
