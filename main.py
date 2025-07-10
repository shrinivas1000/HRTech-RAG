from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List
import logging
import os
import re

# Load environment variables
load_dotenv()

# Global variables
vectordb = None
qa_chain = None

# Request/Response models
class AskRequest(BaseModel):
    query: str

class Source(BaseModel):
    snippet: str
    page: int
    confidence: float

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float

def calculate_confidence(similarity_score: float) -> float:
    """Convert similarity score to confidence percentage"""
    # Normalize similarity score (0-1) to confidence (0.1-0.95)
    return round(0.1 + (similarity_score * 0.85), 2)

def create_snippet(content: str, max_length: int = 200) -> str:
    """Create a snippet from content"""
    return content[:max_length] + "..." if len(content) > max_length else content

def is_source_relevant(query: str, source_content: str, answer: str) -> bool:
    """
    Check if a source is actually relevant to the query and answer.
    Uses keyword matching and context analysis.
    """
    # Extract key terms from query (remove stop words)
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'can', 'i', 'you', 'we', 'they', 'he', 'she', 'it'}
    
    # Clean and extract keywords from query
    query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    query_keywords = query_words - stop_words
    
 
    source_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', source_content.lower()))
    

    dont_know_phrases = ['don\'t know', 'not sure', 'cannot find', 'no information', 'not available', 'unable to answer']
    answer_lower = answer.lower()
    
    if any(phrase in answer_lower for phrase in dont_know_phrases):
        return False
    

    if len(query_keywords) == 0:
        return False
    

    overlap = len(query_keywords.intersection(source_words))
    overlap_ratio = overlap / len(query_keywords) if query_keywords else 0
    

    return overlap_ratio >= 0.3

def extract_answer_keywords(answer: str) -> set:
    """Extract meaningful keywords from the answer"""
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
    answer_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', answer.lower()))
    return answer_words - stop_words

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectordb, qa_chain
    
    try:

        vectordb = Chroma(
            persist_directory="vectorstore",
            embedding_function=OpenAIEmbeddings()
        )
        

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        

        prompt_template = """
        You are an HR assistant. Use the following pieces of context to answer the question.
        If you cannot find relevant information in the context to answer the question, clearly state "I don't have information about this in the HR policies" or similar.
        Do not make up answers. Only use information from the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
     
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        logging.info("Services initialized successfully")
        yield
        
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        raise

app = FastAPI(
    title="HR RAG API", 
    description="Ask questions about HR policies with intelligent source filtering",
    lifespan=lifespan
)

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        if not qa_chain or not vectordb:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        # Generate answer using QA chain
        result = qa_chain.invoke({"query": request.query})
        answer = result.get("result", "")
        
        if not answer:
            raise HTTPException(status_code=500, detail="Failed to generate answer")
        
  
        docs_with_scores = vectordb.similarity_search_with_score(request.query, k=5)
        

        sources = []
        confidence_scores = []
        

        SIMILARITY_THRESHOLD = 0.5  
        
        for doc, similarity_score in docs_with_scores:
            # First filter: similarity score threshold
            if similarity_score > SIMILARITY_THRESHOLD:  # Skip if not similar enough
                continue
                
            # Second filter: relevance check using keyword matching
            if not is_source_relevant(request.query, doc.page_content, answer):
                continue
                
          
            confidence = calculate_confidence(1 - similarity_score)
            
            source = Source(
                snippet=create_snippet(doc.page_content),
                page=doc.metadata.get("page", 0),
                confidence=confidence
            )
            sources.append(source)
            confidence_scores.append(confidence)
        
        # Calculate overall confidence
        if confidence_scores and sources:
            # Use max confidence with slight bonus for multiple sources
            base_confidence = max(confidence_scores)
            multi_source_bonus = min(0.05, len(sources) * 0.01)
            overall_confidence = min(0.95, base_confidence + multi_source_bonus)
        else:
            overall_confidence = 0.1
        
        return AskResponse(
            answer=answer,
            sources=sources,
            confidence=round(overall_confidence, 2)
        )
        
    except Exception as e:
        logging.error(f"Error in /ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "vectorstore_loaded": vectordb is not None,
        "qa_chain_loaded": qa_chain is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)