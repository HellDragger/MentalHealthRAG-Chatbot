import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import shutil
from tqdm import tqdm

# Constants
DATA_PATH = "../data/processed_data/processed_text"
CHROMA_PATH = "../data/chromaprocessed"
MAX_BATCH_SIZE = 5461  # Maximum batch size allowed by Chroma

# Load documents from .txt files
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

# Split the documents into chunks
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# Save chunks to Chroma in batches
def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize SentenceTransformers model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    hf_embedder = HuggingFaceEmbeddings(model_name=model_name)

    db = None

    # Process and save chunks in batches
    for i in tqdm(range(0, len(chunks), MAX_BATCH_SIZE)):
        batch = chunks[i:i + MAX_BATCH_SIZE]

        # Create a new Chroma database or add to the existing one
        if db is None:
            db = Chroma.from_documents(
                batch,
                hf_embedder,
                persist_directory=CHROMA_PATH
            )
        else:
            db.add_documents(batch)

        # Persist the database after each batch
        db.persist()
        print(f"Saved batch {i // MAX_BATCH_SIZE + 1} to {CHROMA_PATH}.")

    print(f"Total saved chunks: {len(chunks)} to {CHROMA_PATH}.")

# Main function to generate the data store
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

if __name__ == "__main__":
    load_dotenv()
    generate_data_store()
