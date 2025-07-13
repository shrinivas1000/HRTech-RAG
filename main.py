from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any
import logging
import os
import re
import datetime
import numpy as np
from document_manager import DocumentManager
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Basic authentication
security = HTTPBasic()
user_sessions: Dict[str, dict] = {}
query_logs: List[dict] = []

# Utility function to handle numpy type conversion
def ensure_json_serializable(obj: Any) -> Any:
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    return obj

# Pydantic models
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

# Authentication functions
def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)) -> User:
    """Enhanced authentication with better error handling"""
    username = credentials.username
    password = credentials.password
    
    # Admin authentication
    is_admin = (username == "admin" and password == "AdminHR2025")
    
    # Basic validation for non-admin users
    if not is_admin and len(password) < 1:
        raise HTTPException(
            status_code=401,
            detail="Password cannot be empty",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    user = User(username=username, is_admin=is_admin)
    
    # Update user session tracking
    user_sessions[username] = {
        "last_login": datetime.datetime.now(),
        "is_admin": is_admin,
        "query_count": user_sessions.get(username, {}).get("query_count", 0)
    }
    
    return user

def require_admin(user: User = Depends(authenticate_user)) -> User:
    """Dependency that ensures user is admin - eliminates code duplication"""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# Utility functions
def log_user_query(username: str, query: str, answer: str, tag: str, confidence: float):
    """Enhanced query logging with numpy type conversion"""
    try:
        # Ensure confidence is a Python float
        confidence = ensure_json_serializable(confidence)
        
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
    except Exception as e:
        logger.error(f"Failed to log user query: {e}")

def calculate_confidence(similarity_score: float) -> float:
    """Calculate confidence score from similarity with numpy type handling"""
    # Convert numpy types to Python float
    similarity_score = ensure_json_serializable(similarity_score)
    
    result = 0.1 + ((1 - similarity_score) * 0.85)
    return round(float(result), 2)

def normalize_text_for_matching(text: str) -> str:
    """Normalize text to handle common variations"""
    text = text.lower()
    # Handle plural/singular variations
    text = text.replace('leaves', 'leave')
    text = text.replace('days', 'day') 
    text = text.replace('weeks', 'week')
    text = text.replace('hours', 'hour')
    # Handle common synonyms
    text = text.replace('sick days', 'sick leave')
    text = text.replace('vacation days', 'annual leave')
    text = text.replace('time off', 'leave')
    text = text.replace('pto', 'leave')
    return text

def create_intelligent_snippet(content: str, query: str = "", max_length: int = 250) -> str:
    """Create intelligent snippet that shows the most relevant content"""
    if not query:
        return content[:max_length] + "..." if len(content) > max_length else content
    
    query_terms = [term.lower() for term in query.split() if len(term) > 2]
    
    # Split content into sentences
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # Score sentences by relevance to query
    sentence_scores = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for term in query_terms if term in sentence_lower)
        
        # Bonus for exact phrase matches
        if len(query_terms) > 1:
            query_words = query.lower().split()
            for i in range(len(query_words) - 1):
                phrase = f"{query_words[i]} {query_words[i+1]}"
                if phrase in sentence_lower:
                    score += 2
        
        if score > 0:
            sentence_scores.append((score, sentence))
    
    # Use relevant sentences if found
    if sentence_scores:
        sentence_scores.sort(key=lambda x: x[0], reverse=True)
        
        selected_sentences = []
        total_length = 0
        
        for score, sentence in sentence_scores:
            sentence_with_period = sentence + "."
            if total_length + len(sentence_with_period) <= max_length:
                selected_sentences.append(sentence)
                total_length += len(sentence_with_period)
            elif not selected_sentences:
                truncated = sentence[:max_length - 3] + "..."
                selected_sentences.append(truncated)
                break
        
        result = ". ".join(selected_sentences)
        if not result.endswith('.') and not result.endswith('...'):
            result += "."
        
        return result
    
    # Fallback: find best starting position
    content_lower = content.lower()
    best_position = 0
    max_matches = 0
    
    window_size = min(max_length, len(content))
    for i in range(0, len(content) - window_size + 1, 30):
        window = content_lower[i:i + window_size]
        matches = sum(1 for term in query_terms if term in window)
        if matches > max_matches:
            max_matches = matches
            best_position = i
    
    # Extract from best position
    if best_position > 0:
        for j in range(best_position, max(0, best_position - 50), -1):
            if content[j] in ' .\n':
                best_position = j + 1
                break
    
    snippet = content[best_position:best_position + max_length]
    return snippet.strip() + "..." if len(content) > best_position + max_length else snippet.strip()

def determine_query_tag(query: str) -> str:
    """Determine tag based on query keywords"""
    query_lower = query.lower()
    
    tag_patterns = {
        "leave": ["leave", "vacation", "sick", "time off", "holiday", "absent", "pto", "maternity", "paternity", "bereavement"],
        "benefits": ["salary", "pay", "insurance", "health", "dental", "vision", "retirement", "401k", "bonus", "compensation", "benefits", "perks"],
        "work-arrangements": ["remote", "work from home", "flexible", "schedule", "hours", "attendance", "hybrid", "telecommute", "wfh"],
        "performance": ["review", "evaluation", "feedback", "goal", "training", "development", "promotion", "appraisal", "rating"],
        "policies": ["policy", "rule", "regulation", "compliance", "code of conduct", "guideline", "procedure", "protocol", "harassment", "discrimination"],
        "workplace": ["office", "facility", "equipment", "technology", "communication", "space", "environment", "safety"],
        "issues": ["complaint", "grievance", "conflict", "dispute", "problem", "issue", "concern"]
    }
    
    for tag, keywords in tag_patterns.items():
        if any(keyword in query_lower for keyword in keywords):
            return tag
    
    return "miscellaneous"

def is_source_relevant_improved(query: str, source_content: str, answer: str) -> bool:
    """Improved relevance check with tighter HR-specific logic"""
    
    # If answer says "don't know", reject ALL sources
    dont_know_phrases = ['don\'t know', 'don\'t have information', 'no information', 'not available', 'cannot find', 'not sure']
    if any(phrase in answer.lower() for phrase in dont_know_phrases):
        return False
    
    # Skip very short sources
    if len(source_content.strip()) < 50:
        return False
    
    query_lower = query.lower()
    source_lower = source_content.lower()
    
    # Check for actual policy content indicators
    policy_indicators = ['entitled', 'policy', 'days per year', 'certificate required', 'employees are', 'must', 'required']
    has_policy_content = any(indicator in source_lower for indicator in policy_indicators)
    
    # Specific HR topic logic
    if 'sick' in query_lower:
        has_sick = 'sick' in source_lower
        has_leave_context = any(term in source_lower for term in ['sick leave', 'sick day', 'medical certificate', '10 days'])
        return has_sick and has_leave_context and has_policy_content
    
    if 'vacation' in query_lower or 'annual' in query_lower:
        has_vacation = any(term in source_lower for term in ['annual leave', 'vacation', '20 days', '20 working days'])
        return has_vacation and has_policy_content
    
    if 'remote' in query_lower or 'work from home' in query_lower or 'wfh' in query_lower:
        has_remote = any(term in source_lower for term in ['remote', 'work from home', 'wfh', '3 days', 'manager approval'])
        return has_remote and has_policy_content
    
    if 'benefit' in query_lower:
        benefit_terms = ['leave', 'days', 'weeks', 'paid', 'entitled']
        matches = sum(1 for term in benefit_terms if term in source_lower)
        return matches >= 2 and has_policy_content
    
    # General fallback
    query_terms = [word.lower() for word in query.split() if len(word) > 2 and word.lower() not in ['how', 'many', 'what', 'when', 'where', 'does', 'can', 'get']]
    matches = sum(1 for term in query_terms if term in source_lower)
    return matches >= 1 and has_policy_content

def validate_answer_against_sources(answer: str, sources: List[Source], query: str) -> bool:
    """Validate that the answer is actually supported by the sources"""
    
    if not sources:
        uncertainty_phrases = ['don\'t know', 'don\'t have information', 'no information']
        return any(phrase in answer.lower() for phrase in uncertainty_phrases)
    
    # Check if sources are relevant to the query
    query_lower = query.lower()
    query_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_lower))
    
    question_words = {'how', 'what', 'when', 'where', 'why', 'who', 'can', 'should', 'will', 'does', 'get'}
    meaningful_query_terms = query_terms - question_words
    
    relevant_sources_count = 0
    
    for source in sources:
        source_lower = source.snippet.lower()
        source_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', source_lower))
        
        overlap = len(meaningful_query_terms.intersection(source_terms))
        overlap_ratio = overlap / len(meaningful_query_terms) if meaningful_query_terms else 0
        
        if overlap_ratio >= 0.3:
            relevant_sources_count += 1
    
    if relevant_sources_count == 0:
        logger.warning(f"No sources relevant to query '{query}'")
        return False
    
    # Check if answer is supported by sources
    answer_terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', answer.lower()))
    
    source_terms = set()
    for source in sources:
        terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', source.snippet.lower()))
        source_terms.update(terms)
    
    overlap = len(answer_terms.intersection(source_terms))
    overlap_ratio = overlap / len(answer_terms) if answer_terms else 0
    
    if overlap_ratio < 0.3:
        logger.warning(f"Answer not supported by sources (overlap: {overlap_ratio:.2f})")
        return False
    
    # Special validation for emergency queries
    emergency_terms = ['earthquake', 'fire', 'emergency', 'safety', 'evacuation', 'disaster']
    if any(term in query_lower for term in emergency_terms):
        safety_content_found = any(
            any(term in source.snippet.lower() for term in emergency_terms)
            for source in sources
        )
        
        if not safety_content_found:
            logger.warning("Emergency query but no safety content in sources")
            return False
    
    return True

def parse_response_with_tag(full_response: str, query: str) -> tuple[str, str]:
    """Enhanced tag parsing with fallback to query-based tagging"""
    lines = full_response.strip().split('\n')
    
    answer_lines = []
    extracted_tag = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Tag:"):
            tag_part = line.replace("Tag:", "").strip().lower()
            tag_part = re.sub(r'[^\w-]', '', tag_part)
            if tag_part:
                extracted_tag = tag_part
        elif line and not line.startswith("Tag:"):
            answer_lines.append(line)
    
    answer = '\n'.join(answer_lines).strip()
    
    if answer.startswith("Answer:"):
        answer = answer.replace("Answer:", "").strip()
    
    # Validate extracted tag
    allowed_tags = ["leave", "benefits", "work-arrangements", "performance", 
                   "policies", "workplace", "issues", "miscellaneous"]
    
    if extracted_tag and extracted_tag in allowed_tags:
        tag = extracted_tag
    else:
        tag = determine_query_tag(query)
        logger.info(f"Using fallback tag '{tag}' for query: '{query}' (extracted: '{extracted_tag}')")
    
    return answer, tag

async def reinitialize_qa_chain():
    """Reinitialize QA chain after FAISS document processing"""
    global vectordb, qa_chain
    
    try:
        vectordb = document_manager.get_vectorstore()
        
        if vectordb is None:
            logger.warning("No FAISS vectorstore found - QA chain cannot be initialized")
            qa_chain = None
            return
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        prompt_template = """You are an HR assistant. You are NOT smart enough to think beyond the given sources. Use the following pieces of context to answer the question.
If you cannot find relevant information in the context to answer the question, clearly state "I don't have information about this in the HR policies" or similar.
Do not make up answers. Only use information from the provided context. If the question is absurd, and you can't find relevant sources, just say "I DON'T KNOW". That's it!

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

Answer: [Your answer based ONLY on the context above]
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
        
        logger.info("QA chain reinitialized successfully with FAISS vectorstore")
        
    except Exception as e:
        logger.error(f"Failed to reinitialize QA chain: {e}")
        qa_chain = None
        vectordb = None
        raise

# Global variables
vectordb: Optional[FAISS] = None
qa_chain: Optional[RetrievalQA] = None
document_manager = DocumentManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan with proper FAISS initialization"""
    global vectordb, qa_chain
    
    try:
        # Try to load existing FAISS vectorstore
        vectordb = document_manager.get_vectorstore()
        
        if vectordb is not None:
            # Initialize QA chain only if vectorstore exists
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
            prompt_template = """You are an HR assistant. You are NOT smart enough to think beyond the given sources. Use the following pieces of context to answer the question.
If you cannot find relevant information in the context to answer the question, clearly state "I don't have information about this in the HR policies" or similar.
Do not make up answers. Only use information from the provided context. If the question is absurd, and you can't find relevant sources, just say "I DON'T KNOW". That's it!

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

Answer: [Your answer based ONLY on the context above]
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
            
            logger.info("Services initialized successfully with existing FAISS index")
        else:
            logger.warning("No FAISS index found - will be created on first document upload")
            qa_chain = None
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # Don't raise - allow app to start even without vectorstore
        yield

# FastAPI app initialization
app = FastAPI(
    title="HR RAG API", 
    description="Ask questions about HR policies with FAISS-powered semantic search",
    version="2.0.0",
    lifespan=lifespan
)

# Document management endpoints
@app.post("/admin/upload-document")
async def upload_document_to_storage(
    file: UploadFile = File(...),
    admin_user: User = Depends(require_admin)
):
    """Store document and auto-rebuild FAISS embeddings (admin only)"""
    try:
        result = await document_manager.store_uploaded_file(file)
        
        # Reinitialize QA chain after auto-processing
        await reinitialize_qa_chain()
        
        response_data = {
            "message": "Document uploaded and processed successfully",
            "filename": result["filename"],
            "stored_as": result["stored_filename"],
            "file_type": result["file_type"],
            "status": result["status"],
            "auto_processed": result.get("auto_processed", False),
            "file_size": result["file_size"],
            "vectorstore_type": "FAISS"
        }
        
        return ensure_json_serializable(response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload/processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.delete("/admin/documents/{stored_filename}")
async def remove_document(
    stored_filename: str,
    admin_user: User = Depends(require_admin)
):
    """Remove document and auto-rebuild FAISS embeddings (admin only)"""
    success = document_manager.remove_document(stored_filename)
    
    if success:
        # Reinitialize QA chain after auto-processing
        await reinitialize_qa_chain()
        
        response_data = {
            "message": f"Document {stored_filename} removed and FAISS embeddings updated",
            "auto_processed": True,
            "vectorstore_type": "FAISS"
        }
        
        return ensure_json_serializable(response_data)
    else:
        raise HTTPException(status_code=404, detail="Document not found")

@app.post("/admin/process-documents")
async def process_all_documents(admin_user: User = Depends(require_admin)):
    """Process all documents in data/ folder and rebuild FAISS index (admin only)"""
    try:
        result = document_manager.process_all_documents()
        
        # Reinitialize QA chain after processing
        await reinitialize_qa_chain()
        
        response_data = {
            "message": "Document processing completed with FAISS",
            "processed": result["processed_documents"],
            "failed": result["failed_documents"],
            "total_chunks": result["total_chunks"],
            "vectorstore_type": "FAISS",
            "details": result
        }
        
        return ensure_json_serializable(response_data)
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/admin/documents/status")
async def get_document_status(admin_user: User = Depends(require_admin)):
    """Get status of all documents with FAISS index info (admin only)"""
    stats = document_manager.get_document_stats()
    
    response_data = {
        "documents": stats["documents"],
        "summary": {
            "total": stats["total_documents"],
            "stored": stats["status_breakdown"].get("stored", 0),
            "processed": stats["status_breakdown"].get("processed", 0),
            "errors": stats["status_breakdown"].get("error", 0)
        },
        "file_types": stats["file_type_breakdown"],
        "total_size_mb": round(stats["total_size_bytes"] / (1024 * 1024), 2),
        "vectorstore_type": "FAISS",
        "index_exists": stats.get("index_exists", False)
    }
    
    return ensure_json_serializable(response_data)

@app.get("/admin/documents")
async def get_documents(admin_user: User = Depends(require_admin)):
    """Get uploaded documents list with FAISS integration info (admin only)"""
    stats = document_manager.get_document_stats()
    
    response_data = {
        "documents": stats["documents"],
        "total_documents": stats["total_documents"],
        "total_chunks": stats["total_chunks"],
        "status_breakdown": stats["status_breakdown"],
        "file_type_breakdown": stats["file_type_breakdown"],
        "vectorstore_type": "FAISS",
        "index_exists": stats.get("index_exists", False)
    }
    
    return ensure_json_serializable(response_data)

# Query endpoint
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, user: User = Depends(authenticate_user)):
    """Ask questions about HR policies using FAISS-powered semantic search"""
    try:
        if not qa_chain or not vectordb:
            raise HTTPException(
                status_code=503, 
                detail="No documents have been uploaded yet. Please contact your administrator to upload HR policy documents."
            )
        
        result = qa_chain.invoke({"query": request.query})
        full_response = result.get("result", "")
        
        if not full_response:
            raise HTTPException(status_code=500, detail="Failed to generate answer")
        
        answer, tag = parse_response_with_tag(full_response, request.query)
        
        # Get potential sources from FAISS
        docs_with_scores = vectordb.similarity_search_with_score(request.query, k=5)
        
        # Enhanced debug output for FAISS
        logger.info(f"FAISS Query: '{request.query}' - Found {len(docs_with_scores)} potential sources")
        
        sources = []
        confidence_scores = []
        
        for i, (doc, similarity_score) in enumerate(docs_with_scores):
            # Convert numpy float32 to Python float
            similarity_score = ensure_json_serializable(similarity_score)
            
            # Check similarity threshold
            if similarity_score > 0.9:
                continue
                
            # Check relevance
            is_relevant = is_source_relevant_improved(request.query, doc.page_content, answer)
            if not is_relevant:
                continue
            
            # Calculate confidence
            confidence = calculate_confidence(similarity_score)
            if confidence < 0.4:
                continue
                
            # Create intelligent snippet
            snippet = create_intelligent_snippet(doc.page_content, request.query)
            
            source = Source(
                snippet=snippet,
                page=doc.metadata.get("page", 0),
                confidence=confidence
            )
            sources.append(source)
            confidence_scores.append(confidence)
        
        # Validate answer against sources
        is_valid = validate_answer_against_sources(answer, sources, request.query)
        
        if not is_valid:
            answer = "I do not have any knowledge on that, Sorry."
            sources = []
            overall_confidence = 0.1
            tag = "miscellaneous"
        else:
            overall_confidence = max(confidence_scores) if confidence_scores else 0.1
        
        # Ensure all values are JSON serializable
        overall_confidence = ensure_json_serializable(overall_confidence)
        
        log_user_query(user.username, request.query, answer, tag, overall_confidence)
        
        return AskResponse(
            answer=answer,
            sources=sources,
            confidence=overall_confidence,
            tag=tag
        )
        
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Admin dashboard endpoints
@app.get("/admin/dashboard")
async def admin_dashboard(admin_user: User = Depends(require_admin)):
    """Enhanced admin dashboard with FAISS integration info"""
    total_queries = len(query_logs)
    unique_users = len(user_sessions)
    
    tag_counts = {}
    for log in query_logs:
        tag = log["tag"]
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    recent_queries = query_logs[-10:] if query_logs else []
    
    # Get vectorstore status
    stats = document_manager.get_document_stats()
    
    response_data = {
        "total_queries": total_queries,
        "unique_users": unique_users,
        "tag_distribution": tag_counts,
        "recent_queries": recent_queries,
        "active_users": list(user_sessions.keys()),
        "vectorstore_info": {
            "type": "FAISS",
            "index_exists": stats.get("index_exists", False),
            "total_documents": stats["total_documents"],
            "total_chunks": stats["total_chunks"]
        }
    }
    
    return ensure_json_serializable(response_data)

@app.get("/admin/users")
async def get_users(admin_user: User = Depends(require_admin)):
    """Get user information (admin only)"""
    response_data = {
        "users": user_sessions,
        "total_users": len(user_sessions)
    }
    
    return ensure_json_serializable(response_data)

@app.get("/admin/queries")
async def get_all_queries(admin_user: User = Depends(require_admin)):
    """Get all query logs (admin only)"""
    response_data = {
        "queries": query_logs,
        "total_queries": len(query_logs)
    }
    
    return ensure_json_serializable(response_data)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Enhanced health check with FAISS status"""
    stats = document_manager.get_document_stats()
    
    response_data = {
        "status": "healthy",
        "vectorstore_type": "FAISS",
        "vectorstore_loaded": vectordb is not None,
        "qa_chain_loaded": qa_chain is not None,
        "index_exists": stats.get("index_exists", False),
        "total_documents": stats["total_documents"],
        "total_chunks": stats["total_chunks"]
    }
    
    return ensure_json_serializable(response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
