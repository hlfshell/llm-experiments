from llms.gpt import GPT
from llms.tools import Function, Argument

# ai = GPT()

# result = ai.generate("Hi, can you tell me an interesting cat fact? Me-wow!")
# result = ai.generate("Add the following numbers and return nothing else: 1, 2, 4, 5")


def tmp(params):
    print(params)
    print(type(params))
    raise "dead"


weather = Function(
    "weather",
    "Get the current weather for a location",
    [
        Argument(
            "location",
            "The location to get the weather for, like a city or zip code",
            "string",
        ),
        Argument(
            "units",
            "The units to return the temperature in - either Celsius or Fahrenheit",
            "string",
            required=False,
        ),
    ],
    tmp,
)

ai = GPT(tools={"weather": weather})

result = ai.generate("What is the weather like in San Diego, CA right now?")

print(result)
