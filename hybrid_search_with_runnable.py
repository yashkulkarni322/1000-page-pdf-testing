from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import time
import requests
import logging

from langchain.retrievers import EnsembleRetriever
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode #type:ignore
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import VLLMOpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DENSE_EMBEDDING_URL = "http://192.168.1.11:8073/encode_text"
QDRANT_URL = "http://192.168.1.13:6333"
COLLECTION_NAME = "pi_scout_case_docs"
K = 5
LLM_URL = "http://192.168.1.11:8077/v1"
LLM_MODEL = "RedHatAI/gemma-3-27b-it-quantized.w4a16"

app = FastAPI(title="RAG Retriever API")

logger.info("Starting RAG Retriever API initialization...")
logger.info(f"Configuration: Collection={COLLECTION_NAME}, K={K}, LLM={LLM_MODEL}")

class CustomAPIEmbeddings(Embeddings):
    """Custom embeddings class that uses the local embedding API"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        logger.info(f"CustomAPIEmbeddings initialized with URL: {api_url}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents - used by LangChain for validation"""
        logger.info(f"embed_documents called for {len(texts)} texts")
        return [self.embed_query(text) for text in texts]
    
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

logger.info("Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL)
embeddings = CustomAPIEmbeddings(api_url=DENSE_EMBEDDING_URL)

logger.info("Initializing dense retriever...")
dense_vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE,
    vector_name="dense-embed",
    content_payload_key="page_content"
)
dense_retriever = dense_vectorstore.as_retriever(search_kwargs={"k": K})
logger.info("Dense retriever initialized successfully")

logger.info("Initializing sparse retriever...")
sparse_model = FastEmbedSparse(model_name="Qdrant/bm25")
sparse_vectorstore = QdrantVectorStore(
    client=client,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    sparse_embedding=sparse_model,
    retrieval_mode=RetrievalMode.SPARSE,
    sparse_vector_name="sparse-embed",
    content_payload_key="page_content"
)
sparse_retriever = sparse_vectorstore.as_retriever(search_kwargs={"k": K})
logger.info("Sparse retriever initialized successfully")

logger.info("Creating ensemble retriever...")
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)
logger.info("Ensemble retriever created with weights [0.5, 0.5]")

logger.info("Initializing LLM...")
llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=LLM_URL,
    model_name=LLM_MODEL,
    temperature=0.0,
    max_tokens=2048
)
logger.info("LLM initialized successfully")

template = """You are an expert assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

Question: {question} 
Context: {context} 
Answer:
"""

rag_prompt = ChatPromptTemplate.from_template(template)

logger.info("Creating RAG chain...")
rag_chain = (
    RunnableParallel(
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)
logger.info("RAG chain created successfully")

def format_docs(docs):
    return "\n\n".join(f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs))

def extract_sources(docs):
    unique_sources = []
    seen_sources = set()
    
    for doc in docs:
        if hasattr(doc, 'metadata') and 'source_path' in doc.metadata:
            source = doc.metadata.get('source_path') or doc.metadata.get('file_name', 'Unknown')
            
            if source not in seen_sources:
                seen_sources.add(source)
                unique_sources.append(source)
    
    return "\n".join(unique_sources)

logger.info("Retrievers and LLM ready. API is now accepting requests.")

class RAGGenerateRequest(BaseModel):
    query: str

class RAGGenerateResponse(BaseModel):
    query: str
    result: str
    citations: str
    source: str

@app.post("/rag/generate", response_model=RAGGenerateResponse)
async def generate_rag(request: RAGGenerateRequest):
    logger.info(f"Received query: '{request.query[:100]}...'")
    total_start = time.time()

    logger.info("Step 1: Starting document retrieval...")
    retrieval_start = time.time()
    docs = ensemble_retriever.invoke(request.query)
    
    # Limit to exactly K documents
    docs = docs[:K]
    
    retrieval_time = time.time() - retrieval_start
    logger.info(f"Retrieved {len(docs)} documents in {retrieval_time:.3f}s")

    logger.info("Step 2: Formatting context...")
    formatting_start = time.time()
    context_str = format_docs(docs)
    formatting_time = time.time() - formatting_start
    logger.info(f"Context formatted ({len(context_str)} chars) in {formatting_time:.3f}s")

    logger.info("Step 3: Invoking RAG chain...")
    generation_start = time.time()
    answer = rag_chain.invoke({"question": request.query, "context": context_str})
    generation_time = time.time() - generation_start
    logger.info(f"RAG chain generated response (length: {len(answer)} chars)")

    sources = extract_sources(docs)
    
    total_time = time.time() - total_start
    
    logger.info(f"Request completed in {total_time:.3f}s (retrieval: {retrieval_time:.3f}s, formatting: {formatting_time:.3f}s, generation: {generation_time:.3f}s)")

    return RAGGenerateResponse(
        query=request.query,
        result=answer,
        citations=context_str,
        source=sources
    )
