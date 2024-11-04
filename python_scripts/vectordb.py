from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
import chromadb
import nest_asyncio

nest_asyncio.apply()

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Get or create a collection in Chroma
collection = chroma_client.get_or_create_collection(name="mental")

# Specify the local directory containing PDF files
pdf_directory = "../data/raw_data/PDF_Files"

# Load all PDFs from the directory
pdf_loaders = [PyPDFLoader(os.path.join(pdf_directory, pdf_file)) for pdf_file in os.listdir(pdf_directory) if pdf_file.endswith('.pdf')]
docs = []
for loader in pdf_loaders:
    docs.extend(loader.load())

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunked_documents = text_splitter.split_documents(docs)

# Embeddings model for text embedding
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Prepare documents and their embeddings
texts = [doc.page_content for doc in chunked_documents]
ids = [f"doc_{i}" for i in range(len(texts))]
embeddings_vectors = embeddings.embed_documents(texts)

# Ensure consistent embedding dimension
embedding_dimension = len(embeddings_vectors[0])

# Upsert the chunked documents into ChromaDB
collection.upsert(
    documents=texts,
    embeddings=embeddings_vectors,
    ids=ids
)