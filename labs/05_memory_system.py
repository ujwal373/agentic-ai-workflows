import os, json, time
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------- Tools -----------
def get_weather(city):
    return f"The weather in {city} is cloudy and 14°C."

def travel_tip(weather):
    if "rain" in weather or "cloudy" in weather:
        return "Carry a light jacket or umbrella."
    else:
        return "Perfect day for outdoor plans!"

TOOLS = {"get_weather": get_weather, "travel_tip": travel_tip}


# ----------- Memory helpers -----------
MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def summarize_recent(memory, n=3):
    return "\n".join([f"Q: {m['query']} | A: {m['response']}" for m in memory[-n:]])


# ----------- Agent -----------
class Step(BaseModel):
    action: str
    arg: str

def agent(query: str):
    memory = load_memory()
    recent = summarize_recent(memory)
    context = f"Recent interactions:\n{recent}\n" if recent else "No prior memory.\n"

    prompt = f"""
You are a reasoning agent with memory.
Use your past experiences if helpful.

{context}

User query: {query}

Tools available: get_weather(city), travel_tip(weather_description)
Return JSON {{"action":"tool_name","arg":"value"}}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content.strip()
    print("Model output:", content)

    try:
        step = Step(**json.loads(content))
    except Exception as e:
        return {"error": f"Parsing failed: {e}", "raw": content}

    if step.action not in TOOLS:
        return {"error": f"Invalid tool '{step.action}'"}

    result = TOOLS[step.action](step.arg)

    # ---- Save memory ----
    memory.append({"query": query, "response": result})
    save_memory(memory)

    return {"response": result, "memory_size": len(memory)}


if __name__ == "__main__":
    print(json.dumps(agent("What should I pack for Dublin today?"), indent=2))
