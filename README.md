# 📚 RAG Research Assistant

An AI-powered Research Assistant that allows users to upload PDF documents and ask questions about their content. It uses Retrieval-Augmented Generation (RAG), semantic search, FAISS, Sentence Transformers and Groq LLM inference to generate context-grounded answers with source references.

## 🚀 Features

- PDF text extraction using PyMuPDF
- Text cleaning and overlapping chunking
- Semantic embeddings using `all-MiniLM-L6-v2`
- FAISS vector database for similarity search
- Dynamic PDF uploads through Streamlit
- Incremental vector index updates
- Conversational RAG with chat history
- Question rewriting for contextual follow-up questions
- Source citations with document names and page numbers
- Groq-powered answer generation
- Retrieval evaluation using Recall@3 and MRR
- Answer evaluation using keyword coverage and faithfulness
- Input and PDF validation

## 🏗️ Architecture

```text
PDF Documents
     |
     v
PDF Text Extraction
     |
     v
Text Cleaning and Chunking
     |
     v
Sentence Transformer Embeddings
     |
     v
FAISS Vector Database
     |
     v
User Question
     |
     v
Semantic Retrieval
     |
     v
Retrieved Context
     |
     v
Groq LLM
     |
     v
Grounded Answer + Sources
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.12 |
| PDF Processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| LLM Inference | Groq API |
| Interface | Streamlit |
| Data Processing | Pandas, NumPy |

## 📂 Project Structure

```text
rag-research-assistant/
│
├── app/
│   ├── pdf_extractor.py
│   ├── text_processor.py
│   ├── embedding_generator.py
│   ├── vector_store.py
│   ├── document_ingestion.py
│   ├── llm_generator.py
│   ├── rag_pipeline.py
│   ├── conversation.py
│   ├── evaluation.py
│   └── answer_evaluation.py
│
├── data/
│   ├── pdfs/
│   ├── extracted/
│   ├── embeddings/
│   ├── vector_store/
│   └── evaluation/
│
├── tests/
│   ├── test_pdf_upload.py
│   ├── test_pdf_extractor.py
│   ├── test_dynamic_chunking.py
│   ├── test_document_ingestion.py
│   ├── test_invalid_pdfs.py
│   ├── test_input_validation.py
│   ├── test_groq.py
│   ├── test_conversation.py
│   └── test_converastional_rag.py
│
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🧪 Testing

The project includes individual test scripts for:

- PDF extraction and upload
- Dynamic text chunking
- Document ingestion and FAISS updates
- Invalid and corrupted PDF handling
- User input validation
- Groq API integration
- Conversational functionality
- Conversational RAG pipeline

Run individual tests from the project root:

```bash
python -m tests.test_pdf_upload
python -m tests.test_pdf_extractor
python -m tests.test_dynamic_chunking
python -m tests.test_document_ingestion
python -m tests.test_invalid_pdfs
python -m tests.test_input_validation
python -m tests.test_groq
python -m tests.test_conversation
python -m tests.test_converastional_rag
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd rag-research-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_actual_api_key
```

Never commit your `.env` file.

### 5. Run the application

```bash
streamlit run streamlit_app.py
```

## 📊 Evaluation

### Retrieval Evaluation

- Recall@3
- Mean Reciprocal Rank (MRR)

### Answer Evaluation

- Keyword Coverage
- LLM-as-a-Judge Faithfulness
- Average Response Time

Run the evaluation:

```bash
python -m app.evaluation
python -m app.answer_evaluation
```

Evaluation results are saved in `data/evaluation/`.

## 🔮 Future Improvements

- OCR support for scanned PDFs
- Improved retrieval using hybrid search
- Reranking retrieved chunks
- Expanded evaluation dataset
- Persistent storage for uploaded documents
- Automated deployment and monitoring

## 👨‍💻 Author

Deepak Kumar Mahato

AI/ML Engineer | RAG | NLP | Deep Learning
