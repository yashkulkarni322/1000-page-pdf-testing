from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import time
import requests
import logging

from langchain.retrievers import EnsembleRetriever
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode #type:ignore
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Config ===
DENSE_EMBEDDING_URL = "http://192.168.1.11:8073/encode_text"
QDRANT_URL = "http://192.168.1.13:6333"
COLLECTION_NAME = "1000_pages_no_overlap"
K = 10
LLM_URL = "http://192.168.1.11:8077/v1/chat/completions"
LLM_MODEL = "RedHatAI/gemma-3-27b-it-quantized.w4a16"

app = FastAPI(title="RAG Retriever API")

logger.info("Starting RAG Retriever API initialization...")
logger.info(f"Configuration: Collection={COLLECTION_NAME}, K={K}, LLM={LLM_MODEL}")

# === Custom Embeddings Class for API ===
class CustomAPIEmbeddings(Embeddings):
    """Custom embeddings class that uses the local embedding API"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        logger.info(f"CustomAPIEmbeddings initialized with URL: {api_url}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        logger.info(f"Embedding {len(texts)} documents...")
        embeddings = []
        for idx, text in enumerate(texts):
            embedding = self.embed_query(text)
            embeddings.append(embedding)
            if (idx + 1) % 10 == 0:
                logger.info(f"Embedded {idx + 1}/{len(texts)} documents")
        logger.info(f"Successfully embedded all {len(texts)} documents")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            logger.debug(f"Embedding query (length: {len(text)} chars)")
            payload = {'text': text}
            response = requests.post(
                self.api_url,
                headers={
                    'accept': 'application/json',
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                data=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    logger.debug(f"Query embedded successfully (embedding dim: {len(result['embeddings'])})")
                    return result['embeddings']
                else:
                    logger.error(f"Embedding API error: {result}")
                    raise RuntimeError(f"Embedding API error: {result}")
            else:
                logger.error(f"HTTP Error {response.status_code} from embedding API")
                raise RuntimeError(f"HTTP Error {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error calling embedding API: {str(e)}")
            raise RuntimeError(f"Error calling embedding API: {str(e)}")

# === Qdrant client & vectorstores ===
logger.info("Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL)
embeddings = CustomAPIEmbeddings(api_url=DENSE_EMBEDDING_URL)

# Dense retriever
logger.info("Initializing dense retriever...")
dense_vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE,
    vector_name="dense",
    content_payload_key="text"
)
dense_retriever = dense_vectorstore.as_retriever(search_kwargs={"k": K})
logger.info("Dense retriever initialized successfully")

# Sparse retriever
logger.info("Initializing sparse retriever...")
sparse_model = FastEmbedSparse(model_name="Qdrant/bm25")
sparse_vectorstore = QdrantVectorStore(
    client=client,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    sparse_embedding=sparse_model,
    retrieval_mode=RetrievalMode.SPARSE,
    sparse_vector_name="sparse",
    content_payload_key="text"
)
sparse_retriever = sparse_vectorstore.as_retriever(search_kwargs={"k": K})
logger.info("Sparse retriever initialized successfully")

# Ensemble retriever
logger.info("Creating ensemble retriever...")
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)
logger.info("Ensemble retriever created with weights [0.5, 0.5]")

# === Prompt Template ===
template = """You are an expert assistant. Use the context below to reason step by step before giving a final answer. Think logically and only answer based on the provided information.

Context:
{context}

Question: {question}

Think step by step, then provide your final answer clearly marked as: "Answer: <final answer>"."""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

logger.info("Retrievers and LLM ready. API is now accepting requests.")

# === API Schema ===
class RAGGenerateRequest(BaseModel):
    query: str

class RAGGenerateResponse(BaseModel):
    answer: str
    context: List[str]
    retrieval_time: float
    formatting_time: float
    generation_time: float
    total_time: float

# === Endpoint ===
@app.post("/rag/generate", response_model=RAGGenerateResponse)
async def generate_rag(request: RAGGenerateRequest):
    logger.info(f"Received query: '{request.query[:100]}...'")
    total_start = time.time()

    # Step 1: Retrieve documents
    logger.info("Step 1: Starting document retrieval...")
    retrieval_start = time.time()
    docs = ensemble_retriever.invoke(request.query)
    retrieval_time = time.time() - retrieval_start
    logger.info(f"Retrieved {len(docs)} documents in {retrieval_time:.3f}s")

    # Step 2: Format context
    logger.info("Step 2: Formatting context...")
    formatting_start = time.time()
    context_str = format_docs(docs)
    formatting_time = time.time() - formatting_start
    logger.info(f"Context formatted ({len(context_str)} chars) in {formatting_time:.3f}s")

    # Step 3: Format prompt
    logger.info("Step 3: Formatting prompt...")
    formatted_prompt = prompt.invoke({
        "context": context_str,
        "question": request.query
    }).to_string()
    logger.info(f"Prompt formatted (total length: {len(formatted_prompt)} chars)")

    # Step 4: Call LLM
    logger.info("Step 4: Calling LLM...")
    generation_start = time.time()
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": formatted_prompt}
        ]
    }
    response = requests.post(
        LLM_URL,
        headers={"Content-Type": "application/json"},
        json=payload
    )
    if response.status_code != 200:
        logger.error(f"LLM returned error: {response.status_code} - {response.text}")
        raise RuntimeError(f"LLM returned error: {response.status_code} - {response.text}")
    
    data = response.json()
    if "choices" in data and len(data["choices"]) > 0:
        answer = data["choices"][0]["message"]["content"].strip()
        logger.info(f"LLM generated response (length: {len(answer)} chars)")
    else:
        answer = "No valid response from model."
        logger.warning("LLM returned no valid response")

    generation_time = time.time() - generation_start
    total_time = time.time() - total_start
    
    logger.info(f"Request completed in {total_time:.3f}s (retrieval: {retrieval_time:.3f}s, formatting: {formatting_time:.3f}s, generation: {generation_time:.3f}s)")

    return RAGGenerateResponse(
        answer=answer,
        context=[doc.page_content for doc in docs],
        retrieval_time=retrieval_time,
        formatting_time=formatting_time,
        generation_time=generation_time,
        total_time=total_time
    )