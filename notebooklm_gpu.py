from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import docx2txt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from datetime import datetime
import numpy as np

# --- RAG Specific Imports ---
from sentence_transformers import SentenceTransformer
import faiss

# --- MODIFIED: Define the compute device globally ---
# This will check if a CUDA-enabled GPU is available and use it; otherwise, it will fall back to the CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Global variables for Models, Data, and RAG ---
llm_model = None
llm_tokenizer = None
embedding_model = None
documents = {}
text_chunks = {}
faiss_indices = {}
chat_history = []

# --- System Prompts for the AI Model (UPDATED) ---
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. Use the following documents to answer the user's question. Answer the question based only on the information provided in the documents. If the answer cannot be found in the documents, say so.
**Important: Always respond in the same language as the user's question.**"""

MULTI_DOC_SYSTEM_PROMPT = """You are a helpful AI assistant specializing in synthesizing information from multiple sources. The user has selected multiple documents and is asking a general question.

Your task is to follow these instructions carefully:
1.  **Synthesize First:** Begin your response with a concise introductory paragraph that summarizes the main themes of the documents and how they relate to each other.
2.  **Detailed Breakdown:** After the introduction, provide a more detailed breakdown. Address each document separately, using its filename as a clear reference point.
3.  **Extract Key Points:** For each document, summarize its key points, concepts, or stages as presented in the provided context.
4.  **Base Answers on Context:** Your entire response MUST be based ONLY on the text provided in the 'context' section.
5.  **Important: Always respond in the same language as the user's question.**
"""


def load_models():
    """Load both the Llama 3.2 LLM and the embedding model onto the specified device."""
    global llm_model, llm_tokenizer, embedding_model
    # <-- MODIFIED: Inform the user which device is being used.
    print(f"Loading AI models... This may take a few minutes. Using device: {device}")

    # --- Load LLM (Llama 3.2) ---
    llm_model_name = "./llama_local" 
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            # <-- MODIFIED: Use float16 for better GPU memory efficiency.
            torch_dtype=torch.float16,
            # <-- MODIFIED: Automatically map the model to the GPU if available.
            device_map=device,
            low_cpu_mem_usage=True
        )
        print("LLM model loaded successfully!")
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        llm_model, llm_tokenizer = None, None

    # --- Load Embedding Model (for RAG) ---
    try:
        # <-- MODIFIED: Load the SentenceTransformer model directly onto the target device.
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("Embedding model loaded successfully!")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        embedding_model = None

# --- Text Extraction Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text

def extract_text_from_docx(filepath):
    return docx2txt.process(filepath)

def extract_text_from_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(filepath, filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'pdf': return extract_text_from_pdf(filepath)
    elif ext == 'docx': return extract_text_from_docx(filepath)
    elif ext == 'txt': return extract_text_from_txt(filepath)
    return ""

# --- RAG Functions ---
def chunk_text(text, chunk_size=512, chunk_overlap=64):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_faiss_index(chunks, model):
    if not chunks or model is None: return None
    # Embeddings are generated on the GPU for speed.
    embeddings = model.encode(chunks, convert_to_tensor=True)
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    # The Faiss index is on the CPU, so we move embeddings to the CPU before adding.
    index.add(embeddings.cpu().numpy())
    return index

def generate_response(question, context, system_prompt):
    """Generate response using Llama model with a specific system prompt."""
    if llm_model is None or llm_tokenizer is None:
        return "Model not loaded. Please restart the server."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the context from the documents:\n\n{context}\n\n---\n\nQuestion: {question}"}
    ]
    try:
        prompt = llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # <-- MODIFIED: Move tokenized inputs to the same device as the model (GPU).
        inputs = llm_tokenizer(
            prompt, return_tensors="pt", max_length=4096, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=[llm_tokenizer.eos_token_id, llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            )
        
        response_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = llm_tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response if response else "I couldn't generate a response."
    except Exception as e:
        print(f"Error during response generation: {e}")
        return f"Error generating response: {str(e)}"

# --- API Endpoints ---
@app.route('/')
def index():
    return send_from_directory('.', 'version3.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type or no file selected'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        text = extract_text(filepath, filename)
        file_id = str(datetime.now().timestamp()) # Use timestamp for unique ID
        documents[file_id] = {'filename': filename, 'text': text}
        chunks = chunk_text(text)
        text_chunks[file_id] = chunks
        faiss_indices[file_id] = create_faiss_index(chunks, embedding_model)
        
        return jsonify({'success': True, 'file_id': file_id, 'filename': filename})
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message', '')
    selected_file_ids = data.get('file_ids', [])
    
    if not question: return jsonify({'error': 'No message provided'}), 400
    if not selected_file_ids: return jsonify({'response': 'Please select at least one document.'})

    # <-- MODIFIED: Encode on GPU, then move to CPU and convert to NumPy for Faiss search.
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    question_embedding_np = question_embedding.cpu().numpy()
    question_embedding_np = np.expand_dims(question_embedding_np, axis=0)

    retrieved_chunks, unique_contents = [], set()
    for file_id in selected_file_ids:
        if file_id in faiss_indices and faiss_indices[file_id]:
            index = faiss_indices[file_id]
            k = min(5, len(text_chunks.get(file_id, [])))
            if k == 0: continue
            
            # <-- MODIFIED: Use the NumPy array for searching in the CPU-based Faiss index.
            distances, indices = index.search(question_embedding_np, k=k)
            for i in indices[0]:
                if i >= 0:
                    chunk_content = text_chunks[file_id][i]
                    if chunk_content not in unique_contents:
                        unique_contents.add(chunk_content)
                        retrieved_chunks.append({
                            "source": documents[file_id]['filename'],
                            "content": chunk_content
                        })

    if not retrieved_chunks:
        return jsonify({'response': "I couldn't find relevant information.", 'sources': []})

    context = "\n\n".join([f"Source: {chunk['source']}\nContent: {chunk['content']}" for chunk in retrieved_chunks])
    
    system_prompt_to_use = DEFAULT_SYSTEM_PROMPT
    general_keywords = ['summarize', 'summary', 'compare', 'contrast', 'overview', 'about', 'explain', 'what are', 'ملخص', 'مقارنة', 'اشرح', 'ما هي', 'ماذا تتحدث']
    
    if len(selected_file_ids) > 1 and any(keyword in question.lower() for keyword in general_keywords):
        system_prompt_to_use = MULTI_DOC_SYSTEM_PROMPT
        print("Using multi-document synthesis prompt.")

    response = generate_response(question, context, system_prompt_to_use)
    chat_history.append({'question': question, 'answer': response})
    
    return jsonify({'response': response, 'sources': retrieved_chunks})

@app.route('/files', methods=['GET'])
def get_files():
    return jsonify({'files': [{'id': fid, 'filename': data['filename']} for fid, data in documents.items()]})

@app.route('/files/<file_id>', methods=['DELETE'])
def delete_file(file_id):
    if file_id in documents:
        filename = documents[file_id]['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath): os.remove(filepath)
        del documents[file_id]
        if file_id in text_chunks: del text_chunks[file_id]
        if file_id in faiss_indices: del faiss_indices[file_id]
        return jsonify({'success': True})
    return jsonify({'error': 'File not found'}), 404

@app.route('/history', methods=['DELETE'])
def clear_history():
    global chat_history
    chat_history = []
    return jsonify({'success': True})

if __name__ == '__main__':
    print("Starting NotebookLM Clone...")
    load_models()
    print("\nServer starting at http://localhost:5000")
    app.run(debug=False, port=5000)