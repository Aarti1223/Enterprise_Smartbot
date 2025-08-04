import os
import json
import sqlite3
from uuid import uuid4
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import pickle

# ----------------- Load .env ------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in .env file.")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file.")

# ----------------- Token Handling ------------------
def count_tokens_approx(text: str) -> int:
    return len(text) // 4

def log_token_usage(query: str, response: str):
    tokens_input = count_tokens_approx(query)
    tokens_output = count_tokens_approx(response)
    total_tokens = tokens_input + tokens_output
    log_entry = {
        "query": query,
        "response": response,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "total_tokens": total_tokens
    }
    with open("token_usage_log.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"🧒 Token usage - In: {tokens_input}, Out: {tokens_output}, Total: {total_tokens}")
    return tokens_input, tokens_output

# ----------------- Database ------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS user_tokens (user_id TEXT PRIMARY KEY, token_limit INT, token_used INT, token_remaining INT)")
    c.execute("INSERT OR IGNORE INTO user_tokens VALUES (?, ?, ?, ?)", ("user1", 10000, 0, 10000))
    c.execute("CREATE TABLE IF NOT EXISTS history (user_id TEXT, query TEXT, response TEXT, input_tokens INT, output_tokens INT, total_tokens INT)")
    c.execute("""CREATE TABLE IF NOT EXISTS feedback (
        user_id TEXT,
        original_query TEXT,
        original_response TEXT,
        corrected_answer TEXT
    )""")
    conn.commit()
    conn.close()

def get_user_tokens(user_id):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT token_limit, token_used, token_remaining FROM user_tokens WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row if row else (0, 0, 0)

def update_user_tokens(user_id, tokens_used):
    token_limit, used, remaining = get_user_tokens(user_id)
    new_used = used + tokens_used
    new_remaining = max(0, remaining - tokens_used)
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("UPDATE user_tokens SET token_used = ?, token_remaining = ? WHERE user_id = ?", 
              (new_used, new_remaining, user_id))
    conn.commit()
    conn.close()

def save_history(user_id, query, response, input_tokens, output_tokens):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?, ?)", 
              (user_id, query, response, input_tokens, output_tokens, input_tokens + output_tokens))
    conn.commit()
    conn.close()

def save_feedback(user_id, original_query, original_response, corrected_answer):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO feedback VALUES (?, ?, ?, ?)", 
              (user_id, original_query, original_response, corrected_answer))
    conn.commit()
    conn.close()

def get_corrected_feedback(user_id, query):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT corrected_answer FROM feedback WHERE user_id = ? AND original_query = ?", (user_id, query))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

# ----------------- Pinecone & GPT Setup ------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "enterprise-gen-ai-smartbot"

if index_name not in [index.name for index in pc.list_indexes()]:
    raise ValueError(f"❌ Pinecone index '{index_name}' not found. Please create it first.")

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

# ----------------- Main Logic ------------------
USER_ID = "user1"

def ask_user_question(query):
    input_tokens = count_tokens_approx(query)
    _, _, remaining = get_user_tokens(USER_ID)
    print(f"🔢 Estimated input tokens: {input_tokens}")
    print(f"💰 Tokens remaining for user '{USER_ID}': {remaining}")
    if remaining < input_tokens + 50:
        print("❌ Not enough tokens. Please recharge to continue.")
        return None

    corrected = get_corrected_feedback(USER_ID, query)
    if corrected:
        output_text = corrected
        print("📌 Using corrected feedback from user.")
    else:
        response = rag_chain.invoke({"input": query})
        output_text = response["answer"]

    output_tokens = count_tokens_approx(output_text)
    total_tokens = input_tokens + output_tokens
    log_token_usage(query, output_text)
    save_history(USER_ID, query, output_text, input_tokens, output_tokens)
    update_user_tokens(USER_ID, total_tokens)
    print("🤖 Answer:", output_text)
    print(f"✅ Tokens used: {total_tokens} (Input: {input_tokens}, Output: {output_tokens})")
    return output_text

def feedback_rag_loop():
    while True:
        query = input("\n🔎 Ask something (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = ask_user_question(query)
        if not answer:
            continue
        feedback = input("❓ Is the answer correct? (yes/no): ").strip().lower()
        if feedback == "no":
            correct_answer = input("📝 Please correct the answer: ")
            corrected_embedding = embedding_model.embed_query(correct_answer)
            vector_id = f"feedback-{str(uuid4())}"
            metadata = {"query": query, "corrected": True, "text": correct_answer}
            index.upsert([(vector_id, corrected_embedding, metadata)])
            save_feedback(USER_ID, query, answer, correct_answer)
            print("✅ ✅ Final Answer (Updated):", correct_answer)
        else:
            print("👍 Great! Moving to next query.")

def upload_chunks_to_pinecone(pkl_path="chunks.pkl", flag_file="upload_done.flag"):
    if os.path.exists(flag_file):
        print("✅ Chunks already uploaded to Pinecone. Skipping re-upload.")
        return
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"❌ Missing: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)

    vectors = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("full_text", "").strip()
        if not text:
            continue

        embedding = embedding_model.embed_query(text)
        vector_id = f"chunk-{i}"
        metadata = {
            "source": "docx_chunk",
            "type": chunk.get("type", "unknown"),
            "text": text
        }
        vectors.append((vector_id, embedding, metadata))

    # Upload in batches
    batch_size = 100
    for i in tqdm(range(0, len(vectors), batch_size), desc="📤 Uploading chunks to Pinecone"):
        batch = vectors[i:i + batch_size]
        index.upsert(batch)
        
    with open(flag_file, "w") as f:
        f.write("upload complete")
    print(f"✅ Uploaded {len(vectors)} chunks to Pinecone.")

if __name__ == "__main__":
    init_db()
    upload_chunks_to_pinecone()
    feedback_rag_loop()
