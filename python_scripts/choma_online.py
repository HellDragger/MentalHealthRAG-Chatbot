import os
import json
import pandas as pd
import PyPDF2
import torch
from sentence_transformers import SentenceTransformer
from chromadb import Client
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_dpr_response(query, collection, tokenizer, model):
    results = collection.query(
        query=model.encode(query, convert_to_tensor=True),
        n=5,
    )
    context = " ".join([result["metadatas"][0]["query"] for result in results])
    inputs = tokenizer(query, context, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_data_from_pdf(file_path, model):
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        embeddings = [model.encode(text, convert_to_tensor=True)]
        metadatas = [{"file_path": file_path}]
        return embeddings, metadatas

def load_data_from_csv(file_path, model, query_column, context_column):
    df = pd.read_csv(file_path)
    embeddings = []
    metadatas = []
    for index, row in df.iterrows():
        query = row[query_column]
        context = row[context_column]
        embeddings.append(model.encode(context, convert_to_tensor=True))
        metadatas.append({"query": query, "file_path": file_path})
    return embeddings, metadatas

def load_data_from_json(file_path, model, query_key, context_key):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        embeddings = []
        metadatas = []
        for item in data:
            query = item[query_key]
            context = item[context_key]
            embeddings.append(model.encode(context, convert_to_tensor=True))
            metadatas.append({"query": query, "file_path": file_path})
        return embeddings, metadatas

def main():
    # Load embedding model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load Chroma DB client
    client = Client(host="localhost", port=6656)

    # Create a new collection for DPR
    collection = client.create_collection(
        name="rag_collection",
        embedding_function=lambda x: model.encode(x, convert_to_tensor=True),
    )

    # Load data from PDF files
    pdf_folder = '../data/PDF_Files'
    for file_name in os.listdir(pdf_folder):
        file_path = os.path.join(pdf_folder, file_name)
        embeddings, metadatas = load_data_from_pdf(file_path, model)
        collection.add(embeddings=embeddings, metadatas=metadatas)

    # Load data from CSV files
    csv_folder = '../data/CSV_Files'
    for file_name in os.listdir(csv_folder):
        file_path = os.path.join(csv_folder, file_name)
        embeddings, metadatas = load_data_from_csv(file_path, model, 'query', 'context')
        collection.add(embeddings=embeddings, metadatas=metadatas)

    # Load data from JSON files
    json_folder = '../data/JSON_Files'
    for file_name in os.listdir(json_folder):
        file_path = os.path.join(json_folder, file_name)
        embeddings, metadatas = load_data_from_json(file_path, model, 'query', 'context')
        collection.add(embeddings=embeddings, metadatas=metadatas)

    # Load RAG model
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    rag_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    # Example usage
    query = "What is the capital of France?"
    response = generate_dpr_response(query, collection, tokenizer, rag_model)
    print(response)

if __name__ == "__main__":
    main()