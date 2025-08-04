tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                        "default": "celsius"
                    },
                },
                "required": ["location","format"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                        "default": "celsius"
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                        "default": 1
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
]


from openai import OpenAI
openai_api_key = "None"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

query = """What's the weather like today in San Francisco"""

chat_response = client.chat.completions.create(
    model="Hammer2/Hammer2.0-0.5b",
    messages=[
        {"role": "user", "content": query},],
    tools = tools,
    temperature=0
)

# Output result
msg = chat_response.choices[0].message

print("Function Name:", msg.function_call.name if msg.function_call else "N/A")
print("Function Arguments:", msg.function_call.arguments if msg.function_call else "N/A")

print(chat_response.choices[0].message.content)