import os, json, random, time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Tools ----------
def get_weather(city):
    return f"The weather in {city} is {random.choice(['cloudy', 'rainy', 'sunny'])}."

def travel_tip(weather):
    if "rain" in weather:
        return "Bring an umbrella."
    elif "cloudy" in weather:
        return "Pack a light jacket."
    else:
        return "Enjoy the sunshine!"

def tell_joke(_):
    return random.choice([
        "Why did the model cross the road? To fine-tune itself!",
        "I told my code a joke, but it didn’t get the reference error."
    ])

TOOLS = {"get_weather": get_weather, "travel_tip": travel_tip, "tell_joke": tell_joke}


# ---------- Control agent ----------
def agent(query, max_loops=3):
    history = []
    result = None

    for step in range(max_loops):
        # --- build reasoning prompt ---
        prompt = f"""
You are an autonomous agent orchestrator.
User query: {query}
Previous context: {result or "none"}

Available tools: get_weather(city), travel_tip(weather_desc), tell_joke()
Decide your next action in JSON: {{"tool": "...", "arg": "...", "done": true/false}}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip()
        print(f"\n🧭 Step {step+1} -> {raw}")

        try:
            plan = json.loads(raw)
        except Exception:
            history.append({"error": "JSON parse failed", "raw": raw})
            break

        tool, arg, done = plan.get("tool"), plan.get("arg", ""), plan.get("done", False)

        if tool not in TOOLS:
            history.append({"error": f"Invalid tool '{tool}'"})
            break

        try:
            result = TOOLS[tool](arg)
            history.append({"step": step+1, "tool": tool, "arg": arg, "result": result})
        except Exception as e:
            history.append({"error": str(e)})
            break

        if done:
            break

    return {"history": history, "final": result}


if __name__ == "__main__":
    print(json.dumps(agent("I'm traveling to Dublin tomorrow, any advice?"), indent=2))
