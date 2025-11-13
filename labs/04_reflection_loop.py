import os, json, time
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Tools ----
def get_weather(city):
    return f"The weather in {city} is cloudy and 14°C."

def travel_tip(weather):
    if "rain" in weather or "cloudy" in weather:
        return "Carry a light jacket or umbrella."
    else:
        return "Perfect day for outdoor plans!"

TOOLS = {"get_weather": get_weather, "travel_tip": travel_tip}


# ---- Model schema ----
class Step(BaseModel):
    action: str
    arg: str


def agent(query: str):
    steps = []
    reflection = ""
    context = ""

    for i in range(2):  # allow up to 2 reasoning rounds
        prompt = f"""
You are a reflective agent.
User query: {query}
Previous context: {context}
Reflection: {reflection}

Tools available: get_weather(city), travel_tip(weather_description)
Plan your next step as JSON: {{"action":"tool_name","arg":"value"}}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content.strip()
        print(f"\n🧠 Round {i+1} model output ->", content)

        try:
            step = Step(**json.loads(content))
        except Exception as e:
            return {"error": f"Parsing failed at step {i+1}: {e}", "raw": content}

        if step.action not in TOOLS:
            reflection = f"Invalid tool '{step.action}'. Ending."
            break

        result = TOOLS[step.action](step.arg)
        reflection = f"Tool '{step.action}' returned: {result}"
        steps.append({"step": i+1, "action": step.action, "arg": step.arg, "result": result})
        context = result

    return {"steps": steps, "final_reflection": reflection}


if __name__ == "__main__":
    print(json.dumps(agent("I’m planning a trip to Dublin, any suggestion?"), indent=2))
