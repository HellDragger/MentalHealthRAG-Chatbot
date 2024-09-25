from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline

# Constants
CHROMA_PATH = "../data/chroma"  # Replace with your Chroma directory path

# Prompt template for RAG
PROMPT_TEMPLATE = """
Answer the following question based only on the provided context:
{context}
- - -
Question: {question}
Answer:
"""

# Evaluation thresholds (adjust based on your requirements)
RELEVANCE_THRESHOLD = 0.7

# Function to evaluate embeddings
def evaluate_embeddings(results, query_text):
    if len(results) == 0:
        print("No results found for the query.")
        return False
    relevance_scores = [score for _, score in results]
    if all(score < RELEVANCE_THRESHOLD for score in relevance_scores):
        print(f"Low relevance scores: {relevance_scores}")
        return False
    print(f"Embeddings passed the evaluation. Relevance scores: {relevance_scores}")
    return True

# Function to evaluate generated response
def evaluate_generated_response(response_text, expected_keywords):
    if not response_text:
        print("No response generated.")
        return False
    # Evaluate based on expected keywords or length
    for keyword in expected_keywords:
        if keyword.lower() not in response_text.lower():
            print(f"Keyword '{keyword}' not found in the response.")
            return False
    print(f"Generated response passed the evaluation: {response_text}")
    return True

# Function to evaluate sources
def evaluate_sources(sources):
    if len(sources) == 0:
        print("No sources found.")
        return False
    for source in sources:
        if source is None:
            print(f"Missing source information in some documents.")
            return False
    print(f"Sources passed the evaluation: {sources}")
    return True

# Overall evaluation function
def evaluate_rag_system(query_text, expected_keywords):
    try:
        # Embedding function
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_function = HuggingFaceEmbeddings(model_name=model_name)

        # Prepare the database
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Retrieving the context from the DB using similarity search
        results = db.similarity_search_with_relevance_scores(query_text, k=3)

        # Step 1: Evaluate embeddings
        if not evaluate_embeddings(results, query_text):
            return None, None

        # Combine context from matching documents
        context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])

        # Create prompt template using context and query text
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Initialize GPT-2 model with adjusted configuration for better output
        model = pipeline("text-generation", model="gpt2", max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9)

        # Step 2: Generate response text
        response_text = model(prompt)[0]['generated_text'].strip()
        response_text = response_text.split("Answer:", 1)[-1].strip()  # Post-process to remove unwanted text

        # Step 3: Evaluate generated response
        if not evaluate_generated_response(response_text, expected_keywords):
            return None, None

        # Step 4: Evaluate sources
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        if not evaluate_sources(sources):
            return None, None

        # Format and return the response including generated text and sources
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return formatted_response, response_text

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

def main():
    # Query text and expected keywords for evaluation
    query_text = "What is OCD?"
    expected_keywords = ["OCD"]  # Example of expected keywords

    # Call the evaluation function
    formatted_response, response_text = evaluate_rag_system(query_text, expected_keywords)

    # Print the final response
    if response_text:
        print(f"Formatted Response:\n{formatted_response}")
    else:
        print("No response generated.")

if __name__ == "__main__":
    main()
