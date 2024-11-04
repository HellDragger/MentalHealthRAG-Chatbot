import os
import shutil
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from vllm import LLM
from vllm.sampling_params import SamplingParams
import faiss

# Constants
DATA_PATH = "../data/processed_data/text"
CHROMA_PATH = "../data/chroma"
MAX_BATCH_SIZE = 5461
MISTRAL_MODEL = "mistralai/Mistral-8B-Instruct-2410"
SAMPLING_PARAMS = SamplingParams(max_tokens=8192)

# Step 1: Load documents from .txt files
def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                doc = Document(page_content=text, metadata={"source": filename})
                documents.append(doc)
    print(f"Loaded {len(documents)} documents from {DATA_PATH}.")
    return documents

# Step 2: Split the documents into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, length_function=len, add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# Step 3: Save chunks to Chroma in batches
def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize MiniLM embedder
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    hf_embedder = HuggingFaceEmbeddings(model_name=model_name)

    db = None
    for i in tqdm(range(0, len(chunks), MAX_BATCH_SIZE)):
        batch = chunks[i:i + MAX_BATCH_SIZE]
        if db is None:
            db = Chroma.from_documents(batch, hf_embedder, persist_directory=CHROMA_PATH)
        else:
            db.add_documents(batch)
        db.persist()
        print(f"Saved batch {i // MAX_BATCH_SIZE + 1} to {CHROMA_PATH}.")
    return db

# Step 4: Build FAISS index for retrieval
def build_faiss_index(chunks):
    embeddings = [chunk.page_content for chunk in chunks]
    index = faiss.IndexFlatL2(len(embeddings[0]))  # Assuming embeddings are fixed-length
    index.add(embeddings)
    faiss.write_index(index, "faiss_index")
    return FAISS(index, chunks)

# Step 5: Mistral generator
def generate_answer(query, retriever):
    llm = LLM(model=MISTRAL_MODEL, tokenizer_mode="mistral", config_format="mistral", load_format="mistral")
    retriever_qa = RetrievalQA(llm=llm, retriever=retriever, prompt_template="<s>[INST]{query}[/INST]{answer}</s>[INST]{new_query}[/INST]")
    response = retriever_qa.run(query=query)
    return response

# Main function to run the RAG pipeline
def main():
    # Load and process documents
    documents = load_documents()
    chunks = split_text(documents)

    # Embed and save chunks to Chroma
    db = save_to_chroma(chunks)

    # Build FAISS index for retrieval
    retriever = build_faiss_index(chunks)

    # User query input and Mistral response generation
    query = input("Enter your query: ")
    answer = generate_answer(query, retriever)
    print("Generated Answer:", answer)

if __name__ == "__main__":
    load_dotenv()
    main()
