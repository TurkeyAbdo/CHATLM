# Local NotebookLM: Chat with Your Documents

This project is a powerful, locally-hosted application that allows you to chat with your documents using a large language model (LLM). Inspired by Google's NotebookLM, it uses Retrieval-Augmented Generation (RAG) to provide answers based *only* on the content of your uploaded files, ensuring accuracy and privacy.

Your documents and the AI model never leave your machine.

 <!-- **Action Required:** Replace this with a URL to a screenshot of your app -->

## Features

-   **100% Local and Private:** All processing happens on your own computer. No data is sent to external services.
-   **Multi-Format Support:** Upload and chat with `.pdf`, `.docx`, and `.txt` files.
-   **Source-Grounded Answers:** The AI provides answers directly based on the context from your documents, reducing hallucinations.
-   **GPU Acceleration:** Full support for NVIDIA GPUs via CUDA for significantly faster performance in both embedding and response generation.
-   **CPU Fallback:** Automatically runs in a CPU-only mode if a compatible GPU is not detected.
-   **Multi-Document Synthesis:** Ask general questions (like "summarize these") across multiple selected documents to get a synthesized overview.
-   **Simple Web Interface:** Easy-to-use interface for uploading files and managing your chat session.

## How It Works

The application uses a RAG (Retrieval-Augmented Generation) pipeline:

1.  **Document Processing:** When you upload a file, its text is extracted and split into smaller, manageable chunks.
2.  **Embedding & Indexing:** Each text chunk is converted into a numerical representation (embedding) using a `SentenceTransformer` model. These embeddings are stored in a Faiss vector index for efficient searching.
3.  **Retrieval:** When you ask a question, it's also converted into an embedding. The system then searches the Faiss index to find the most relevant text chunks from your documents.
4.  **Generation:** The retrieved text chunks are passed as context to a large language model (like Llama 3.2), which then generates a natural language answer based on that information.

## Setup and Installation

Follow these steps to get the application running on your local machine.

### Step 1: Prerequisites

-   **Python:** Version 3.8 or newer.
-   **Git:** For cloning the repository.
-   **(For GPU Users)**:
    -   An **NVIDIA GPU** with at least 8GB of VRAM recommended.
    -   The latest **NVIDIA drivers**.
    -   A compatible version of the **CUDA Toolkit**. Check the [PyTorch website](https://pytorch.org/get-started/locally/) to see which CUDA version is recommended.
    -   You can verify your setup by running `nvidia-smi` in your terminal.

### Step 2: Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name```

### Step 3: Download the Language Model

This application requires a local language model in the Hugging Face `transformers` format.

1.  Create a directory to store the model:
    ```bash
    mkdir llama_local
    ```
2.  Download a model from the Hugging Face Hub. For example, to download a smaller, optimized version of Llama 3, you can use `git`:
    ```bash
    # Make sure you have Git LFS installed: git lfs install
    git clone https://huggingface.co/unsloth/llama-3-8b-Instruct-GGUF llama_local 
    # Note: The code is currently set up for a transformers model, not GGUF. 
    # A better choice would be:
    # git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf llama_local
    ```
    *Make sure the path in `app.py` (`llm_model_name = "./llama_local"`) matches where you download the model.*

### Step 4: Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 5: Install Dependencies

Choose the installation path that matches your hardware.

#### A) For GPU Users (Recommended)

First, uninstall any existing CPU versions of PyTorch and Faiss to prevent conflicts.

```bash
pip uninstall torch faiss-cpu
```

Next, install the GPU-enabled libraries.

1.  **Install PyTorch for CUDA:** Get the correct command for your system from the [PyTorch website](https://pytorch.org/get-started/locally/). It will look similar to this:
    ```bash
    # Example for CUDA 12.1 - DO NOT COPY PASTE, get the command from the official site!
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
2.  **Install Faiss-GPU:**
    ```bash
    pip install faiss-gpu
    ```
3.  **Install Remaining Dependencies:**
    ```bash
    pip install Flask Flask-Cors Werkzeug PyMuPDF docx2txt transformers sentence-transformers langchain-text-splitters numpy
    ```

#### B) For CPU-Only Users

If you don't have an NVIDIA GPU, install the CPU versions of the libraries:

```bash
pip install torch faiss-cpu Flask Flask-Cors Werkzeug PyMuPDF docx2txt transformers sentence-transformers langchain-text-splitters numpy
```

## How to Run the Application

Once the setup is complete, start the Flask server with this command:

```bash
python app.py
```

You will see output indicating that the models are loading (this may take a few minutes, especially on the first run). Once you see the line `Server starting at http://localhost:5000`, you can open your web browser and navigate to:

**http://localhost:5000**

## How to Use the Interface

1.  **Upload Files:** Click the "Upload" button to select one or more documents (`.pdf`, `.docx`, `.txt`).
2.  **Select Documents:** In the "My Documents" list, check the box next to the file(s) you want to include in the conversation.
3.  **Ask a Question:** Type your question in the chat input at the bottom and press Enter.
4.  **Receive an Answer:** The AI will generate a response based on the content of your selected documents. The sources used to generate the answer will be displayed for reference.
5.  **Manage Files:** You can delete individual files by clicking the trash icon next to them.

## Customization

-   **Changing the LLM:** To use a different language model, download it from Hugging Face and update the `llm_model_name` variable in `app.py`:
    ```python
    # Example for a different model
    llm_model_name = "./path/to/your/new_model" 
    ```
-   **Changing the Embedding Model:** To use a different sentence transformer model, modify this line in `load_models()`:
    ```python
    # Example for a different embedding model
    embedding_model = SentenceTransformer('all-mpnet-base-v2', device=device)
    ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.