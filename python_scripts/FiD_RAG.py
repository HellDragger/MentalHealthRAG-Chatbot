from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from transformers import T5Tokenizer, T5Model

# Constants
CHROMA_PATH = "../data/chroma"  # Replace with your Chroma directory path

# Prompt template for RAG
PROMPT_TEMPLATE = """
Answer the following question based only on the provided context:
{context}
- -
Question: {question}
Answer:
"""

def query_rag(query_text):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and T5-based FiD model.
    
    Args:
        - query_text (str): The text to query the RAG system with.
    
    Returns:
        - formatted_response (str): Formatted response including the generated text and sources.
        - response_text (str): The generated response text.
    """
    try:
        # Embedding function
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_function = HuggingFaceEmbeddings(model_name=model_name)

        # Prepare the database
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Retrieving the context from the DB using similarity search
        results = db.similarity_search_with_relevance_scores(query_text, k=3)

        # Check if there are any matching results or if the relevance score is too low
        if len(results) == 0 or results[0][1] < 0.7:
            print("Unable to find matching results.")
            return None, None

        # Combine context from matching documents
        context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
        
        # Create prompt template using context and query text
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Initialize FiD model with T5
        model_name = "t5-large"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5Model.from_pretrained(model_name)

        # Encode input prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate response using FiD model
        outputs = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
        
        # Decode generated response
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to remove unwanted text and repetitive phrases
        response_text = response_text.split("Answer:", 1)[-1].strip()
        response_text = response_text.split("\n")[0].strip()  # Take only the first line to avoid repetitions
        
        # Get sources of the matching documents
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        
        # Format and return response including generated text and sources
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return formatted_response, response_text

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

def main():
    # Query text
    query_text = "What is anxiety?"
    
    # Call the query function
    formatted_response, response_text = query_rag(query_text)
    
    # Print the final response
    if response_text:
        print(f"Formatted Response:\n{formatted_response}")
    else:
        print("No response generated.")

if __name__ == "__main__":
    main()
