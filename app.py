from flask import Flask, render_template, request, jsonify
from rag import run_rag_chain

app = Flask(__name__)

# Route for rendering the chatbot UI
@app.route('/')
def index():
    return render_template('frontend.html')

# API endpoint to handle chatbot queries
@app.route('/api/query', methods=['POST'])
def query():
    user_question = request.json.get('question')
    
    if not user_question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Get the response from the RAG system
    try:
        answer = run_rag_chain(user_question)
        return jsonify({'response': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
