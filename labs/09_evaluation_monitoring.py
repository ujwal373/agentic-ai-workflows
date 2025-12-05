import os, json, time, random
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Tools ----------
def get_weather(city):
    return f"The weather in {city} is {random.choice(['rainy','cloudy','sunny'])}."

def travel_tip(weather):
    if "rain" in weather:  return "Bring an umbrella."
    if "cloudy" in weather: return "Pack a jacket."
    return "Enjoy the sunshine!"

TOOLS = {"get_weather": get_weather, "travel_tip": travel_tip}


# ---------- Evaluation helpers ----------
LOG_FILE = "agent_log.json"

def log_event(event):
    """Append evaluation record to a JSON log."""
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            try: logs = json.load(f)
            except: logs = []
    logs.append(event)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)


# ---------- Evaluator ----------
def evaluate(goal, steps, result, start_time):
    duration = round(time.time() - start_time, 2)
    success = not any("error" in s for s in steps)
    score = 1.0 if success else 0.3

    reflection_prompt = f"""
You are an evaluator.
Goal: {goal}
Steps: {json.dumps(steps)}
Final result: {result}
In 1-2 sentences, assess what went well and what could improve.
"""
    feedback = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":reflection_prompt}]
    ).choices[0].message.content.strip()

    event = {
        "goal": goal,
        "success": success,
        "score": score,
        "duration_sec": duration,
        "feedback": feedback,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    log_event(event)
    return event


# ---------- Mini agent pipeline ----------
def simple_agent(goal):
    start = time.time()
    steps = []

    # Step 1
    weather = TOOLS["get_weather"]("Dublin")
    steps.append({"tool":"get_weather","result":weather})

    # Step 2
    tip = TOOLS["travel_tip"](weather)
    steps.append({"tool":"travel_tip","result":tip})

    final = f"{weather} {tip}"
    report = evaluate(goal, steps, final, start)
    return {"final": final, "evaluation": report}


if __name__ == "__main__":
    goal = "Plan for a Dublin trip"
    print(json.dumps(simple_agent(goal), indent=2))
