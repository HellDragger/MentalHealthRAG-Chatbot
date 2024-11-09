import transformers
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
import chromadb
import nest_asyncio
import torch

# Apply nested asyncio to avoid runtime issues
nest_asyncio.apply()

# Initialize ChromaDB client and load the collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="mental")

# Check if CUDA (GPU) is available and set device accordingly
device = 0 if torch.cuda.is_available() else -1  # -1 for CPU

# Initialize the text generation pipeline using Mistral-7B-Instruct-v0.3
model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # Using Mistral-7B-Instruct as the model

text_generation_pipeline = transformers.pipeline(
    model=model_name,
    task="text-generation",
    tokenizer=model_name,  # Ensure tokenizer is correctly set
    temperature=0.2,  # Low temperature for more focused outputs
    repetition_penalty=1.1,
    max_new_tokens=500,
    device=device,  # Use GPU if available
    top_p=0.9,  # Sampling parameter for response diversity
)

# Define the LLM with the HuggingFace pipeline
mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create an instruction-based prompt template
prompt_template = """
**Context:**

{context}

**Question:**
{question}

**Answer:**
"""

# Create prompt from the prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create LLM chain
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

# Function to retrieve documents from ChromaDB based on query
def get_relevant_documents(query):
    retriever_results = collection.query(
        query_texts=[query],
        n_results=2  # Number of relevant documents to retrieve
    )
    
    # Assuming 'documents' contains the context
    documents = retriever_results['documents']
    
    # Flatten the list of documents, if needed
    if isinstance(documents, list) and isinstance(documents[0], list):
        documents = [item for sublist in documents for item in sublist]
    
    # Join the documents to create a single context string
    context = " ".join(documents)  # Concatenate all relevant document texts
    return context

# Function to handle RAG chain logic
def run_rag_chain(user_question):
    # Retrieve relevant context from ChromaDB based on user query
    context = get_relevant_documents(user_question)

    # Prepare inputs for the chain
    inputs = {"context": context, "question": user_question}
    
    # Invoke the LLM chain and extract the generated response
    result = llm_chain.invoke(inputs)
    
    # Extract and clean the response text (strip unnecessary characters)
    response_text = result["text"].strip()
    
    return response_text

if __name__ == "__main__":
    # Example: User inputs a question
    user_question = input("Please enter your question: ")
    
    # Run RAG chain with the user's question
    answer = run_rag_chain(user_question)
    
    # Output the result (just the response text)
    print("\nAnswer:")
    print(answer)
