from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from langchain.prompts import ChatPromptTemplate

# Constants
CHROMA_PATH = "../data/chroma"  # Replace with your Chroma directory path
DPR_MODEL_NAME = "facebook/dpr-question_encoder-single-nq-base"
DPR_CONTEXT_MODEL_NAME = "facebook/dpr-ctx_encoder-single-nq-base"
BART_MODEL_NAME = "facebook/bart-large-cnn"

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
    Query a Retrieval-Augmented Generation (RAG) system using DPR and BART.
    
    Args:
        - query_text (str): The text to query the RAG system with.
    
    Returns:
        - formatted_response (str): Formatted response including the generated text and sources.
        - response_text (str): The generated response text.
    """
    try:
        # Initialize DPR question encoder and tokenizer
        question_encoder = DPRQuestionEncoder.from_pretrained(DPR_MODEL_NAME)
        question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(DPR_MODEL_NAME)

        # Initialize DPR context encoder and tokenizer
        context_encoder = DPRContextEncoder.from_pretrained(DPR_CONTEXT_MODEL_NAME)
        context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(DPR_CONTEXT_MODEL_NAME)

        # Tokenize and encode the query
        query_inputs = question_tokenizer(query_text, return_tensors="pt")

        # Generate query embeddings
        query_embedding = question_encoder(**query_inputs).pooler_output

        # For demonstration, we'll assume that the Chroma DB holds the context embeddings
        # This part assumes you already have pre-encoded context vectors from documents in your DB
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=None)  # Load pre-existing Chroma DB

        # Search the database for similar documents
        results = db.similarity_search_with_relevance_scores(query_embedding, k=3)

        # Check if there are any matching results or if the relevance score is too low
        if len(results) == 0 or results[0][1] < 0.7:
            print("Unable to find matching results.")
            return None, None

        # Combine context from matching documents
        context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])

        # Create prompt template using context and query text
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Initialize BART model and tokenizer
        bart_model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME)
        bart_tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)

        # Tokenize the prompt
        inputs = bart_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

        # Generate response using BART
        output_ids = bart_model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
        response_text = bart_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

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
