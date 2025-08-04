from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import json
import sqlite3
from uuid import uuid4
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize AI components
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "enterprise-gen-ai-smartbot"
    index = pc.Index(index_name)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_model
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
    
    system_prompt = (
        "You are an assistant for question-answering tasks.\n"
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    AI_ENABLED = True
except Exception as e:
    print(f"AI components not available: {e}")
    AI_ENABLED = False

# Database initialization
def init_db():
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    
    # Users table
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        name TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    
    # Token management
    c.execute("""CREATE TABLE IF NOT EXISTS user_tokens (
        user_id INTEGER PRIMARY KEY,
        token_limit INTEGER DEFAULT 1000,
        token_used INTEGER DEFAULT 0,
        token_remaining INTEGER DEFAULT 1000,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )""")
    
    # Chat history
    c.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        conversation_id TEXT,
        query TEXT,
        response TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        total_tokens INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )""")
    
    # Feedback/corrections
    c.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        conversation_id TEXT,
        original_query TEXT,
        original_response TEXT,
        corrected_answer TEXT,
        status TEXT DEFAULT 'pending',
        admin_reviewed BOOLEAN DEFAULT FALSE,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )""")
    
    # Token requests
    c.execute("""CREATE TABLE IF NOT EXISTS token_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        requested_tokens INTEGER,
        reason TEXT,
        status TEXT DEFAULT 'pending',
        requested_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        processed_at DATETIME,
        processed_by INTEGER,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (processed_by) REFERENCES users (id)
    )""")
    
    # Create default admin user
    admin_email = "admin@shriram.com"
    admin_password = generate_password_hash("admin123")
    c.execute("INSERT OR IGNORE INTO users (email, password_hash, name, role) VALUES (?, ?, ?, ?)", 
              (admin_email, admin_password, "Admin", "admin"))
    
    # Create default user
    user_email = "john.doe@shriram.com"
    user_password = generate_password_hash("user123")
    c.execute("INSERT OR IGNORE INTO users (email, password_hash, name, role) VALUES (?, ?, ?, ?)", 
              (user_email, user_password, "John Doe", "user"))
    
    # Get user ID and create token record
    c.execute("SELECT id FROM users WHERE email = ?", (user_email,))
    user_result = c.fetchone()
    if user_result:
        user_id = user_result[0]
        c.execute("INSERT OR IGNORE INTO user_tokens (user_id, token_limit, token_used, token_remaining) VALUES (?, ?, ?, ?)", 
                  (user_id, 1000, 250, 750))
    
    conn.commit()
    conn.close()

# Helper functions
def count_tokens_approx(text: str) -> int:
    return len(text) // 4

def get_user_by_email(email):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_tokens(user_id):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("SELECT token_limit, token_used, token_remaining FROM user_tokens WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row if row else (1000, 0, 1000)

def update_user_tokens(user_id, tokens_used):
    token_limit, used, remaining = get_user_tokens(user_id)
    new_used = used + tokens_used
    new_remaining = max(0, remaining - tokens_used)
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("UPDATE user_tokens SET token_used = ?, token_remaining = ? WHERE user_id = ?", 
              (new_used, new_remaining, user_id))
    conn.commit()
    conn.close()

def save_chat_history(user_id, conversation_id, query, response, input_tokens, output_tokens):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("""INSERT INTO chat_history 
                 (user_id, conversation_id, query, response, input_tokens, output_tokens, total_tokens) 
                 VALUES (?, ?, ?, ?, ?, ?, ?)""", 
              (user_id, conversation_id, query, response, input_tokens, output_tokens, input_tokens + output_tokens))
    conn.commit()
    conn.close()

def get_chat_history(user_id, limit=10):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("""SELECT conversation_id, query, response, timestamp 
                 FROM chat_history 
                 WHERE user_id = ? 
                 ORDER BY timestamp DESC 
                 LIMIT ?""", (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return rows

def save_feedback(user_id, conversation_id, original_query, original_response, corrected_answer):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("""INSERT INTO feedback 
                 (user_id, conversation_id, original_query, original_response, corrected_answer) 
                 VALUES (?, ?, ?, ?, ?)""", 
              (user_id, conversation_id, original_query, original_response, corrected_answer))
    conn.commit()
    conn.close()

def create_token_request(user_id, requested_tokens, reason):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("INSERT INTO token_requests (user_id, requested_tokens, reason) VALUES (?, ?, ?)", 
              (user_id, requested_tokens, reason))
    conn.commit()
    conn.close()

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('chat'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        email = data.get('email')
        password = data.get('password')
        
        user = get_user_by_email(email)
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['user_email'] = user[1]
            session['user_name'] = user[3]
            session['user_role'] = user[4]
            
            if request.is_json:
                return jsonify({'success': True, 'redirect': url_for('chat')})
            return redirect(url_for('chat'))
        else:
            if request.is_json:
                return jsonify({'success': False, 'error': 'Invalid credentials'})
            flash('Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/chat')
def chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    token_limit, token_used, token_remaining = get_user_tokens(user_id)
    chat_history = get_chat_history(user_id)
    
    return render_template('chat.html', 
                         user_name=session.get('user_name', 'User'),
                         token_limit=token_limit,
                         token_used=token_used,
                         token_remaining=token_remaining,
                         chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask_question():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    query = data.get('query', '').strip()
    conversation_id = data.get('conversation_id', str(uuid4()))
    
    if not query:
        return jsonify({'error': 'Please enter a question'}), 400
    
    user_id = session['user_id']
    
    # Check tokens
    input_tokens = count_tokens_approx(query)
    _, _, remaining = get_user_tokens(user_id)
    
    if remaining < input_tokens + 50:
        return jsonify({'error': 'Not enough tokens remaining. Please request more tokens.'}), 400
    
    try:
        if AI_ENABLED:
            response = rag_chain.invoke({"input": query})
            output_text = response["answer"]
        else:
            output_text = f"This is a simulated response to your query: '{query}'. The AI system is currently not available."
        
        output_tokens = count_tokens_approx(output_text)
        total_tokens = input_tokens + output_tokens
        
        # Save to history and update tokens
        save_chat_history(user_id, conversation_id, query, output_text, input_tokens, output_tokens)
        update_user_tokens(user_id, total_tokens)
        
        # Get updated token info
        _, token_used, token_remaining = get_user_tokens(user_id)
        
        return jsonify({
            'answer': output_text,
            'conversation_id': conversation_id,
            'tokens_used': total_tokens,
            'tokens_remaining': token_remaining,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        })
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    conversation_id = data.get('conversation_id', '')
    query = data.get('query', '')
    original_response = data.get('original_response', '')
    corrected_answer = data.get('corrected_answer', '')
    
    if not all([conversation_id, query, original_response, corrected_answer]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    user_id = session['user_id']
    
    try:
        save_feedback(user_id, conversation_id, query, original_response, corrected_answer)
        return jsonify({'message': 'Feedback submitted successfully. It will be reviewed by administrators.'})
    except Exception as e:
        return jsonify({'error': f'Failed to submit feedback: {str(e)}'}), 500

@app.route('/request_tokens', methods=['POST'])
def request_tokens():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    requested_tokens = data.get('tokens', 0)
    reason = data.get('reason', '')
    
    if requested_tokens <= 0:
        return jsonify({'error': 'Invalid token amount'}), 400
    
    user_id = session['user_id']
    
    try:
        create_token_request(user_id, requested_tokens, reason)
        return jsonify({'message': 'Token request submitted successfully. It will be reviewed by administrators.'})
    except Exception as e:
        return jsonify({'error': f'Failed to submit request: {str(e)}'}), 500

@app.route('/admin')
def admin_dashboard():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        return redirect(url_for('login'))
    
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    
    # Get pending feedback
    c.execute("""SELECT f.id, u.name, u.email, f.original_query, f.original_response, 
                        f.corrected_answer, f.timestamp
                 FROM feedback f
                 JOIN users u ON f.user_id = u.id
                 WHERE f.status = 'pending'
                 ORDER BY f.timestamp DESC""")
    pending_feedback = c.fetchall()
    
    # Get pending token requests
    c.execute("""SELECT tr.id, u.name, u.email, tr.requested_tokens, tr.reason, tr.requested_at
                 FROM token_requests tr
                 JOIN users u ON tr.user_id = u.id
                 WHERE tr.status = 'pending'
                 ORDER BY tr.requested_at DESC""")
    pending_requests = c.fetchall()
    
    conn.close()
    
    return render_template('admin_dashboard.html', 
                         pending_feedback=pending_feedback,
                         pending_requests=pending_requests)

@app.route('/admin/tokens')
def admin_tokens():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        return redirect(url_for('login'))
    
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    
    # Get all users with token information
    c.execute("""SELECT u.id, u.name, u.email, ut.token_limit, ut.token_used, ut.token_remaining,
                        (SELECT COUNT(*) FROM token_requests WHERE user_id = u.id AND status = 'pending') as pending_requests
                 FROM users u
                 LEFT JOIN user_tokens ut ON u.id = ut.user_id
                 WHERE u.role = 'user'
                 ORDER BY u.name""")
    users = c.fetchall()
    
    # Get pending token requests
    c.execute("""SELECT tr.id, u.name, u.email, tr.requested_tokens, tr.reason, tr.requested_at
                 FROM token_requests tr
                 JOIN users u ON tr.user_id = u.id
                 WHERE tr.status = 'pending'
                 ORDER BY tr.requested_at DESC""")
    pending_requests = c.fetchall()
    
    conn.close()
    
    return render_template('admin_tokens.html', 
                         users=users,
                         pending_requests=pending_requests)

@app.route('/admin/approve_tokens/<int:request_id>', methods=['POST'])
def approve_tokens(request_id):
    if 'user_id' not in session or session.get('user_role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    
    # Get request details
    c.execute("SELECT user_id, requested_tokens FROM token_requests WHERE id = ? AND status = 'pending'", (request_id,))
    request_info = c.fetchone()
    
    if not request_info:
        return jsonify({'error': 'Request not found or already processed'}), 404
    
    user_id, requested_tokens = request_info
    
    # Update user tokens
    token_limit, token_used, token_remaining = get_user_tokens(user_id)
    new_limit = token_limit + requested_tokens
    new_remaining = token_remaining + requested_tokens
    
    c.execute("UPDATE user_tokens SET token_limit = ?, token_remaining = ? WHERE user_id = ?", 
              (new_limit, new_remaining, user_id))
    
    # Update request status
    c.execute("UPDATE token_requests SET status = 'approved', processed_at = CURRENT_TIMESTAMP, processed_by = ? WHERE id = ?", 
              (session['user_id'], request_id))
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Token request approved successfully'})

@app.route('/admin/reject_tokens/<int:request_id>', methods=['POST'])
def reject_tokens(request_id):
    if 'user_id' not in session or session.get('user_role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("UPDATE token_requests SET status = 'rejected', processed_at = CURRENT_TIMESTAMP, processed_by = ? WHERE id = ?", 
              (session['user_id'], request_id))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Token request rejected'})

@app.route('/admin/grant_tokens/<int:user_id>', methods=['POST'])
def grant_tokens(user_id):
    if 'user_id' not in session or session.get('user_role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    tokens_to_grant = data.get('tokens', 0)
    
    if tokens_to_grant <= 0:
        return jsonify({'error': 'Invalid token amount'}), 400
    
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    
    token_limit, token_used, token_remaining = get_user_tokens(user_id)
    new_limit = token_limit + tokens_to_grant
    new_remaining = token_remaining + tokens_to_grant
    
    c.execute("UPDATE user_tokens SET token_limit = ?, token_remaining = ? WHERE user_id = ?", 
              (new_limit, new_remaining, user_id))
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': f'Granted {tokens_to_grant} tokens successfully'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5173)