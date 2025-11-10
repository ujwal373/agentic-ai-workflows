import json, time

TOOLS = {
    "weather": lambda city: f"The weather in {city} is cloudy and 14°C.",
    "time": lambda _: f"The current time is {time.strftime('%H:%M:%S')}.",
}

def agent(message: str):
    thoughts = []
    
    # Step 1: think
    if "weather" in message:
        thoughts.append("User wants weather information.")
        tool = "weather"
        arg = message.split("in")[-1].strip() if "in" in message else "Dublin"
    elif "time" in message:
        thoughts.append("User asked for current time.")
        tool = "time"
        arg = ""
    else:
        thoughts.append("No known intent detected.")
        return {"thoughts": thoughts, "response": "I don’t know that yet."}
    
    # Step 2: act
    thoughts.append(f"Calling tool: {tool} with arg: {arg}")
    result = TOOLS[tool](arg)
    
    # Step 3: observe
    thoughts.append(f"Observed result: {result}")
    
    # Step 4: reflect
    thoughts.append("Returning formatted answer to user.")
    return {"thoughts": thoughts, "response": result}

if __name__ == "__main__":
    query = "What's the weather in Dublin?"
    output = agent(query)
    print(json.dumps(output, indent=2))
