import os, json
from pydantic import BaseModel

class WeatherRequest(BaseModel):
    city: str

def tool_get_weather(req: WeatherRequest):
    return {"city": req.city, "temp_c": 14, "cond": "cloudy"}  # stub

def agent(msg: str):
    # naive parser -> call tool -> format answer
    if "weather" in msg.lower():
        city = msg.split("in")[-1].strip() if "in" in msg else "Dublin"
        data = tool_get_weather(WeatherRequest(city=city))
        return f"{data['city']}: {data['temp_c']}°C, {data['cond']}"
    return "Ask me about the weather in a city."

if __name__ == "__main__":
    print(agent("weather in Dublin"))
