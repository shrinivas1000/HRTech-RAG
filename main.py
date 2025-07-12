from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List, Dict
import logging
import os
import re
import datetime
from fastapi import UploadFile, File
from document_manager import DocumentManager

load_dotenv()

# basic auth
security = HTTPBasic()
user_sessions: Dict[str, dict] = {}
query_logs: List[dict] = []


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
    tag: str

class User(BaseModel):
    username: str
    is_admin: bool = False


def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)) -> User:
    username = credentials.username
    password = credentials.password
    
    is_admin = (username == "admin" and password == "AdminHR2025")
    
    if not is_admin and len(password) < 1:
        raise HTTPException(
            status_code=401,
            detail="Password cannot be empty",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    user = User(username=username, is_admin=is_admin)
    
    user_sessions[username] = {
        "last_login": datetime.datetime.now(),
        "is_admin": is_admin,
        "query_count": user_sessions.get(username, {}).get("query_count", 0)
    }
    
    return user

def log_user_query(username: str, query: str, answer: str, tag: str, confidence: float):
    query_logs.append({
        "username": username,
        "query": query,
        "answer": answer[:100] + "..." if len(answer) > 100 else answer,
        "tag": tag,
        "confidence": confidence,
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    if username in user_sessions:
        user_sessions[username]["query_count"] += 1


def calculate_confidence(similarity_score: float) -> float:
    return round(0.1 + (similarity_score * 0.85), 2)

def create_snippet(content: str, max_length: int = 200) -> str:
    return content[:max_length] + "..." if len(content) > max_length else content

def determine_query_tag(query: str) -> str:
    """
    Determine tag based on query keywords - fallback for inconsistent LLM tagging
    """
    query_lower = query.lower()
    
    # Define keyword patterns for each tag
    tag_patterns = {
        "leave": ["leave", "vacation", "sick", "time off", "holiday", "absent", "pto", "maternity", "paternity", "bereavement"],
        "benefits": ["salary", "pay", "insurance", "health", "dental", "vision", "retirement", "401k", "bonus", "compensation", "benefits", "perks"],
        "work-arrangements": ["remote", "work from home", "flexible", "schedule", "hours", "attendance", "hybrid", "telecommute"],
        "performance": ["review", "evaluation", "feedback", "goal", "training", "development", "promotion", "appraisal", "rating"],
        "policies": ["policy", "rule", "regulation", "compliance", "code of conduct", "guideline", "procedure", "protocol"],
        "workplace": ["office", "facility", "equipment", "technology", "communication", "space", "environment", "safety"],
        "issues": ["complaint", "grievance", "conflict", "dispute", "harassment", "discrimination", "problem", "issue", "concern"]
    }
    
    # Check for matches
    for tag, keywords in tag_patterns.items():
        if any(keyword in query_lower for keyword in keywords):
            return tag
    
    return "miscellaneous"

def is_source_relevant(query: str, source_content: str, answer: str) -> bool:
    """
    Balanced relevance checking - not too strict, not too lenient
    """
    # Check if answer indicates lack of knowledge
    dont_know_phrases = [
        'don\'t know', 'not sure', 'cannot find', 'no information', 
        'not available', 'unable to answer', 'don\'t have information',
        'cannot provide', 'not found', 'unclear', 'uncertain'
    ]
    answer_lower = answer.lower()
    
    if any(phrase in answer_lower for phrase in dont_know_phrases):
        return False
    
    # Normalize text
    query_lower = query.lower().strip()
    source_lower = source_content.lower().strip()
    
    # Skip very short sources that are likely not informative
    if len(source_content.strip()) < 30:  # Reduced from 50
        return False
    
    # Simplified stop words list
    stop_words = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 
        'can', 'i', 'you', 'we', 'they', 'he', 'she', 'it', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'this', 'that'
    }
    
    # Extract query keywords (lowered minimum length)
    query_words = re.findall(r'\b[a-zA-Z]{3,}\b', query_lower)
    query_keywords = [word for word in query_words if word not in stop_words]
    
    # Extract source keywords
    source_words = re.findall(r'\b[a-zA-Z]{3,}\b', source_lower)
    source_keywords = set(word for word in source_words if word not in stop_words)
    
    if not query_keywords:
        return True  # If no keywords, allow it through
    
    # Check for exact phrase matches (highest priority)
    query_phrases = []
    for i in range(len(query_words) - 1):
        phrase = f"{query_words[i]} {query_words[i+1]}"
        query_phrases.append(phrase)
    
    phrase_matches = sum(1 for phrase in query_phrases if phrase in source_lower)
    
    # Check individual keyword matches
    keyword_matches = sum(1 for keyword in query_keywords if keyword in source_keywords)
    
    # If we have phrase matches, that's good enough
    if phrase_matches > 0:
        return True
    
    # Otherwise, check keyword overlap with more lenient threshold
    total_query_terms = len(query_keywords)
    keyword_score = keyword_matches / total_query_terms
    
    # Require at least 25% keyword overlap OR at least 1 keyword match for short queries
    return keyword_score >= 0.25 or (len(query_keywords) <= 2 and keyword_matches >= 1)

def parse_response_with_tag(full_response: str, query: str) -> tuple[str, str]:
    """
    Enhanced tag parsing with fallback to query-based tagging
    """
    lines = full_response.strip().split('\n')
    
    answer_lines = []
    extracted_tag = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Tag:"):
            # Extract tag
            tag_part = line.replace("Tag:", "").strip().lower()
            # Clean the tag
            tag_part = re.sub(r'[^\w-]', '', tag_part)
            if tag_part:
                extracted_tag = tag_part
        elif line and not line.startswith("Tag:"):
            answer_lines.append(line)
    
    # Join answer lines and clean up
    answer = '\n'.join(answer_lines).strip()
    
    # Clean up answer if it starts with "Answer:"
    if answer.startswith("Answer:"):
        answer = answer.replace("Answer:", "").strip()
    
    # Validate extracted tag
    allowed_tags = ["leave", "benefits", "work-arrangements", "performance", 
                   "policies", "workplace", "issues", "miscellaneous"]
    
    # Use extracted tag if valid, otherwise determine from query
    if extracted_tag and extracted_tag in allowed_tags:
        tag = extracted_tag
    else:
        # Fallback to query-based tagging
        tag = determine_query_tag(query)
        logging.info(f"Using fallback tag '{tag}' for query: '{query}' (extracted: '{extracted_tag}')")
    
    return answer, tag


vectordb = None
qa_chain = None

document_manager = DocumentManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectordb, qa_chain
    
    try:
        vectordb = Chroma(
            persist_directory="vectorstore",
            embedding_function=OpenAIEmbeddings()
        )
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        # More explicit prompt for consistent tagging
        prompt_template = """You are an HR assistant. Use the following pieces of context to answer the question.
If you cannot find relevant information in the context to answer the question, clearly state "I don't have information about this in the HR policies" or similar.
Do not make up answers. Only use information from the provided context.

IMPORTANT: You must end your response with exactly one tag line in this format:
Tag: [tag_name]

Choose exactly ONE tag from these options:
- leave (for: vacation, sick leave, parental leave, time off, holidays, PTO)
- benefits (for: salary, health insurance, retirement, perks, compensation, bonuses)
- work-arrangements (for: remote work, flexible hours, attendance, schedules, hybrid work)
- performance (for: reviews, feedback, goals, training, development, evaluations)
- policies (for: code of conduct, compliance, safety, legal requirements, rules)
- workplace (for: facilities, technology, communication, office space, environment)
- issues (for: grievances, complaints, conflict resolution, harassment, disputes)
- miscellaneous (for: any other HR topics not covered above)

Context:
{context}

Question: {question}

Answer: [Your detailed answer here]
Tag: [exactly one tag from the list above]"""
        
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
    description="Ask questions about HR policies with simple authentication",
    lifespan=lifespan
)

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(authenticate_user)
):
    """Upload PDF document (admin only)"""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        result = await document_manager.process_uploaded_file(file)
        
        return {
            "message": "Document uploaded successfully",
            "filename": result.get("filename", "unknown"),
            "chunks_created": result.get("chunk_count", 0)
        }
    except Exception as e:
        logging.error(f"Upload error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/admin/documents")
async def get_documents(user: User = Depends(authenticate_user)):
    """Get uploaded documents list (admin only)"""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return document_manager.get_document_stats()

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, user: User = Depends(authenticate_user)):
    try:
        if not qa_chain or not vectordb:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        result = qa_chain.invoke({"query": request.query})
        full_response = result.get("result", "")
        
        if not full_response:
            raise HTTPException(status_code=500, detail="Failed to generate answer")
        
        answer, tag = parse_response_with_tag(full_response, request.query)
        
        docs_with_scores = vectordb.similarity_search_with_score(request.query, k=5)
        
        sources = []
        confidence_scores = []
        
        SIMILARITY_THRESHOLD = 0.6  # Relaxed threshold (higher score = less similar)
        
        for doc, similarity_score in docs_with_scores:
            # Skip documents with low similarity
            if similarity_score > SIMILARITY_THRESHOLD:
                continue
            
            # Use balanced relevance checking
            if not is_source_relevant(request.query, doc.page_content, answer):
                continue
                
            confidence = calculate_confidence(1 - similarity_score)
            
            # More reasonable minimum confidence threshold
            if confidence < 0.5:
                continue
                
            source = Source(
                snippet=create_snippet(doc.page_content),
                page=doc.metadata.get("page", 0),
                confidence=confidence
            )
            sources.append(source)
            confidence_scores.append(confidence)
        
        # Calculate overall confidence
        if confidence_scores and sources:
            base_confidence = max(confidence_scores)
            multi_source_bonus = min(0.05, len(sources) * 0.01)
            overall_confidence = min(0.95, base_confidence + multi_source_bonus)
        else:
            overall_confidence = 0.1
        
        log_user_query(user.username, request.query, answer, tag, round(overall_confidence, 2))
        
        return AskResponse(
            answer=answer,
            sources=sources,
            confidence=round(overall_confidence, 2),
            tag=tag
        )
        
    except Exception as e:
        logging.error(f"Error in /ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/admin/dashboard")
async def admin_dashboard(user: User = Depends(authenticate_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    total_queries = len(query_logs)
    unique_users = len(user_sessions)
    
    tag_counts = {}
    for log in query_logs:
        tag = log["tag"]
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    recent_queries = query_logs[-10:] if query_logs else []
    
    return {
        "total_queries": total_queries,
        "unique_users": unique_users,
        "tag_distribution": tag_counts,
        "recent_queries": recent_queries,
        "active_users": list(user_sessions.keys())
    }

@app.get("/admin/users")
async def get_users(user: User = Depends(authenticate_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "users": user_sessions,
        "total_users": len(user_sessions)
    }

@app.get("/admin/queries")
async def get_all_queries(user: User = Depends(authenticate_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "queries": query_logs,
        "total_queries": len(query_logs)
    }

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