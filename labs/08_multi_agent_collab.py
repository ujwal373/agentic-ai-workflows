import os, json, random
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Tools ----------
def get_weather(city):
    return f"The weather in {city} is {random.choice(['rainy','sunny','cloudy'])}."

def travel_tip(weather):
    if "rain" in weather:
        return "Pack an umbrella."
    elif "cloudy" in weather:
        return "Bring a light jacket."
    else:
        return "Perfect day for sightseeing!"

TOOLS = {"get_weather": get_weather, "travel_tip": travel_tip}


# ---------- Agent Roles ----------
def planner_agent(goal):
    """Break down a goal into sub-tasks for the executor."""
    prompt = f"""
You are the Planner Agent.
User goal: {goal}

Break this goal into 1–3 clear steps an Executor can perform,
choosing from: get_weather(city), travel_tip(weather).
Return JSON list of steps like:
[{{"tool":"get_weather","arg":"Dublin"}},{{"tool":"travel_tip","arg":"rainy"}}]
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return []


def executor_agent(steps):
    """Execute each step using local tools."""
    results = []
    for s in steps:
        tool, arg = s.get("tool"), s.get("arg", "")
        if tool in TOOLS:
            result = TOOLS[tool](arg)
            results.append({"tool": tool, "arg": arg, "result": result})
        else:
            results.append({"tool": tool, "error": "unknown tool"})
    return results


def manager_agent(goal):
    """Top-level orchestrator coordinating Planner + Executor."""
    plan = planner_agent(goal)
    print("\n🧠 Planner output:", json.dumps(plan, indent=2))
    if not plan:
        return {"error": "planner failed"}

    results = executor_agent(plan)
    summary_prompt = f"""
You are the Manager Agent summarizing Executor results.
Goal: {goal}
Results: {json.dumps(results)}
Compose a final human-readable answer.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}],
    )
    return {
        "plan": plan,
        "results": results,
        "final": resp.choices[0].message.content.strip()
    }


if __name__ == "__main__":
    goal = "I’m visiting Dublin tomorrow, tell me what to expect and what to pack."
    print(json.dumps(manager_agent(goal), indent=2))
