import os, json, faiss, numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"

MEMORY_FILE = "semantic_memory.json"
FAISS_INDEX = "semantic_index.faiss"


# ---------- Helper Functions ----------
def get_embedding(text):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype="float32")


def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_memory(memories):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)


def load_index(dim=1536):
    if os.path.exists(FAISS_INDEX):
        return faiss.read_index(FAISS_INDEX)
    return faiss.IndexFlatL2(dim)


def save_index(index):
    faiss.write_index(index, FAISS_INDEX)


# ---------- Add to memory ----------
def add_memory(query, response):
    emb = get_embedding(response)
    memories = load_memory()
    index = load_index(len(emb))

    # store in FAISS
    index.add(np.array([emb]))
    save_index(index)

    # store metadata
    memories.append({"query": query, "response": response})
    save_memory(memories)


# ---------- Semantic recall ----------
def recall(query, k=2):
    emb = get_embedding(query)
    index = load_index(len(emb))
    memories = load_memory()

    if len(memories) == 0 or index.ntotal == 0:
        return []

    distances, indices = index.search(np.array([emb]), k)
    retrieved = []
    for i in indices[0]:
        if 0 <= i < len(memories):
            retrieved.append(memories[i])
    return retrieved


# ---------- Agent ----------
def agent(query):
    relevant = recall(query)
    context = "\n".join([f"Q: {m['query']} | A: {m['response']}" for m in relevant])

    prompt = f"""
You are an intelligent agent with semantic memory.
Relevant past context:
{context if context else "None"}

User query: {query}
Respond thoughtfully using context if relevant.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = resp.choices[0].message.content.strip()
    add_memory(query, answer)
    return {"response": answer, "memory_recalled": len(relevant)}


if __name__ == "__main__":
    print(json.dumps(agent("How’s Dublin weather today?"), indent=2))
