import os, json, time
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# ---- setup ----
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- define tools ----
def tool_weather(city):
    return f"The weather in {city} is cloudy and 14°C."

def tool_time(_):
    return f"The current time is {time.strftime('%H:%M:%S')}."

TOOLS = {"weather": tool_weather, "time": tool_time}

# ---- schema for model output ----
class ToolCall(BaseModel):
    tool: str
    arg: str

# ---- agent ----
def agent(message: str):
    prompt = f"""
You are a reasoning agent.
Available tools: weather(city), time().
Decide which tool to call and what argument to pass.

User: {message}

Return a JSON object like:
{{"tool":"weather","arg":"Dublin"}}
If none apply, respond {{"tool":"none","arg":""}}.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content.strip()
    print("Model output:", content)
    
    try:
        call = ToolCall(**json.loads(content))
    except Exception as e:
        return {"error": f"Parsing failed: {e}", "raw": content}

    if call.tool in TOOLS:
        result = TOOLS[call.tool](call.arg)
        return {"thoughts": f"Model chose {call.tool}", "response": result}
    else:
        return {"response": "No suitable tool found."}

if __name__ == "__main__":
    print(json.dumps(agent("What’s the weather in Dublin?"), indent=2))
