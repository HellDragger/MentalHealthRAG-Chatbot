import transformers
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
import chromadb
import nest_asyncio

# Apply nested asyncio to avoid runtime issues
nest_asyncio.apply()

# Initialize ChromaDB client and load the collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="mental")

# Initialize the text generation pipeline using GPT-2
model = "gpt2"  # Using GPT-2 as the model

text_generation_pipeline = transformers.pipeline(
    model=model,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,  # Set to False to get just the response text
    max_new_tokens=500,
)

# Create a prompt template
prompt_template = """
**Prompt:**

{context}

**Question:**
{question}
"""

# Define the LLM with the HuggingFace pipeline
gpt2_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from the prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create LLM chain
llm_chain = LLMChain(llm=gpt2_llm, prompt=prompt)

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
