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
    return round(0.1 + ((1 - similarity_score) * 0.85), 2)

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
    
    # Split content into sentences (handle multiple sentence endings)
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # Score sentences by relevance to query
    sentence_scores = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for term in query_terms if term in sentence_lower)
        
        # Bonus for exact phrase matches
        query_lower = query.lower()
        if len(query_terms) > 1:
            # Check for 2-word phrases
            query_words = query.lower().split()
            for i in range(len(query_words) - 1):
                phrase = f"{query_words[i]} {query_words[i+1]}"
                if phrase in sentence_lower:
                    score += 2  # Bonus for phrase match
        
        if score > 0:
            sentence_scores.append((score, sentence))
    
    # If we found relevant sentences, use them
    if sentence_scores:
        # Sort by relevance score
        sentence_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Take the best sentences that fit within max_length
        selected_sentences = []
        total_length = 0
        
        for score, sentence in sentence_scores:
            sentence_with_period = sentence + "."
            if total_length + len(sentence_with_period) <= max_length:
                selected_sentences.append(sentence)
                total_length += len(sentence_with_period)
            elif not selected_sentences:  # Always include at least one sentence
                # Truncate the sentence if it's the first one and too long
                truncated = sentence[:max_length - 3] + "..."
                selected_sentences.append(truncated)
                break
        
        result = ". ".join(selected_sentences)
        if not result.endswith('.') and not result.endswith('...'):
            result += "."
        
        return result
    
    # Fallback: try to find a good starting position
    content_lower = content.lower()
    best_position = 0
    max_matches = 0
    
    # Look for the position with most query term matches
    window_size = min(max_length, len(content))
    for i in range(0, len(content) - window_size + 1, 30):
        window = content_lower[i:i + window_size]
        matches = sum(1 for term in query_terms if term in window)
        if matches > max_matches:
            max_matches = matches
            best_position = i
    
    # Extract from best position, try to start at word boundary
    if best_position > 0:
        # Look backwards for a good starting point (space, period, etc.)
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
    
    # Tightened sick leave logic
    if 'sick' in query_lower:
        has_sick = 'sick' in source_lower
        has_leave_context = any(term in source_lower for term in ['sick leave', 'sick day', 'medical certificate', '10 days'])
        return has_sick and has_leave_context and has_policy_content
    
    # Tightened vacation logic
    if 'vacation' in query_lower or 'annual' in query_lower:
        has_vacation = any(term in source_lower for term in ['annual leave', 'vacation', '20 days', '20 working days'])
        return has_vacation and has_policy_content
    
    # Tightened remote work logic
    if 'remote' in query_lower or 'work from home' in query_lower or 'wfh' in query_lower:
        has_remote = any(term in source_lower for term in ['remote', 'work from home', 'wfh', '3 days', 'manager approval'])
        return has_remote and has_policy_content
    
    # For benefits queries, require multiple policy terms
    if 'benefit' in query_lower:
        benefit_terms = ['leave', 'days', 'weeks', 'paid', 'entitled']
        matches = sum(1 for term in benefit_terms if term in source_lower)
        return matches >= 2 and has_policy_content
    
    # General fallback - require policy content
    query_terms = [word.lower() for word in query.split() if len(word) > 2 and word.lower() not in ['how', 'many', 'what', 'when', 'where', 'does', 'can', 'get']]
    matches = sum(1 for term in query_terms if term in source_lower)
    return matches >= 1 and has_policy_content

def validate_answer_against_sources(answer: str, sources: List[Source], query: str) -> bool:
    """Validate that the answer is actually supported by the sources AND sources are relevant to query"""
    
    # If no sources, answer should be "I don't know"
    if not sources:
        uncertainty_phrases = ['don\'t know', 'don\'t have information', 'no information']
        return any(phrase in answer.lower() for phrase in uncertainty_phrases)
    
    # STEP 1: Check if sources are actually relevant to the query
    query_lower = query.lower()
    query_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_lower))
    
    # Remove common question words
    question_words = {'how', 'what', 'when', 'where', 'why', 'who', 'can', 'should', 'will', 'does', 'get'}
    meaningful_query_terms = query_terms - question_words
    
    relevant_sources_count = 0
    
    for source in sources:
        source_lower = source.snippet.lower()
        source_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', source_lower))
        
        # Check overlap between query and this source
        overlap = len(meaningful_query_terms.intersection(source_terms))
        overlap_ratio = overlap / len(meaningful_query_terms) if meaningful_query_terms else 0
        
        # Source is relevant if it has decent overlap with query terms
        if overlap_ratio >= 0.3:  # At least 30% of query terms should appear in source
            relevant_sources_count += 1
    
    # If no sources are actually relevant to the query, it's invalid
    if relevant_sources_count == 0:
        print(f"🚨 VALIDATION FAILED: No sources relevant to query '{query}'")
        return False
    
    # STEP 2: Check if answer is supported by the sources
    answer_terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', answer.lower()))
    
    # Extract terms from all sources
    source_terms = set()
    for source in sources:
        terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', source.snippet.lower()))
        source_terms.update(terms)
    
    # Check overlap between answer and sources
    overlap = len(answer_terms.intersection(source_terms))
    overlap_ratio = overlap / len(answer_terms) if answer_terms else 0
    
    # If answer has <30% overlap with sources, it's likely hallucinated
    if overlap_ratio < 0.3:
        print(f"🚨 VALIDATION FAILED: Answer not supported by sources (overlap: {overlap_ratio:.2f})")
        return False
    
    # STEP 3: Special validation for specific topics
    # For safety/emergency queries, require explicit safety content in sources
    emergency_terms = ['earthquake', 'fire', 'emergency', 'safety', 'evacuation', 'disaster']
    if any(term in query_lower for term in emergency_terms):
        safety_content_found = False
        for source in sources:
            if any(term in source.snippet.lower() for term in emergency_terms):
                safety_content_found = True
                break
        
        if not safety_content_found:
            print(f"🚨 VALIDATION FAILED: Emergency query but no safety content in sources")
            return False
    
    print(f"✅ VALIDATION PASSED: Answer supported by relevant sources")
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
    
    # Clean up answer if it starts with "Answer:"
    if answer.startswith("Answer:"):
        answer = answer.replace("Answer:", "").strip()
    
    # Validate extracted tag
    allowed_tags = ["leave", "benefits", "work-arrangements", "performance", 
                   "policies", "workplace", "issues", "miscellaneous"]
    
    if extracted_tag and extracted_tag in allowed_tags:
        tag = extracted_tag
    else:
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
        
        # Enhanced prompt with stronger constraints
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
        
        # Get potential sources with PERMANENT DEBUGGING
        docs_with_scores = vectordb.similarity_search_with_score(request.query, k=5)
        
        # PERMANENT DEBUG OUTPUT
        print(f"\n=== HR RAG DEBUG SESSION ===")
        print(f"Query: '{request.query}'")
        print(f"Found {len(docs_with_scores)} potential sources from vector search")
        print(f"Answer generated: {answer[:100]}...")
        print(f"Tag assigned: {tag}")
        
        sources = []
        confidence_scores = []
        
        for i, (doc, similarity_score) in enumerate(docs_with_scores):
            print(f"\n📄 --- Source {i+1} Analysis ---")
            print(f"📊 Similarity Score: {similarity_score:.4f}")
            print(f"📖 Content Preview: {doc.page_content[:150]}...")
            print(f"📍 Page: {doc.metadata.get('page', 'Unknown')}")
            
            # Check similarity threshold - more lenient now
            if similarity_score > 0.9:  # Changed from 0.8 to 0.9 for more sources
                print("  ❌ REJECTED: Similarity score too high (> 0.9)")
                continue
            else:
                print("  ✅ PASSED: Similarity check")
                
            # Check relevance with improved logic
            is_relevant = is_source_relevant_improved(request.query, doc.page_content, answer)
            if not is_relevant:
                print("  ❌ REJECTED: Failed improved relevance check")
                continue
            else:
                print("  ✅ PASSED: Improved relevance check")
            
            # Calculate confidence
            confidence = calculate_confidence(similarity_score)
            print(f"Calculated confidence: {confidence}")
            
            # Check confidence threshold - more lenient
            if confidence < 0.4:  # Changed from 0.5 to 0.4
                print("  ❌ REJECTED: Confidence too low (< 0.4)")
                continue
            else:
                print("  ✅ PASSED: Confidence check")
                
            # Create intelligent snippet
            snippet = create_intelligent_snippet(doc.page_content, request.query)
            print(f"  📝 Generated snippet: {snippet[:100]}...")
            
            source = Source(
                snippet=snippet,
                page=doc.metadata.get("page", 0),
                confidence=confidence
            )
            sources.append(source)
            confidence_scores.append(confidence)
            print("  🎯 ACCEPTED AS SOURCE")
        
        print(f"\nPreliminary result: {len(sources)} sources accepted")
        print(f"Confidence scores: {confidence_scores}")
        
        # CRITICAL: Validate answer against sources to prevent hallucinations
        is_valid = validate_answer_against_sources(answer, sources, request.query)
        
        if not is_valid:
            print("🚨 HALLUCINATION/IRRELEVANT SOURCES DETECTED - Overriding with 'I DON'T KNOW'")
            answer = "I do not have any knowledge on that, Sorry."
            sources = []
            overall_confidence = 0.1
            tag = "miscellaneous"
        else:
            # Calculate overall confidence
            if confidence_scores and sources:
                overall_confidence = max(confidence_scores)
            else:
                overall_confidence = 0.1
        
        print(f"Final result after validation: {len(sources)} sources, confidence: {overall_confidence}")
        
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
