import os
from typing import Any, Dict, List, Optional, Union

import openai

from llms.agent import LLM, Agent, FunctionCalls
from llms.openai import OpenAIGPT
from llms.tools import Argument, Function

gpt = OpenAIGPT()


def crasher(*args):
    print("Crasher:", args)
    raise "crash"


def news(*args):
    print("News:", args)
    return "No news"


def weather(*args):
    print("Weather:", args)
    return "Sunny and pristine :-)"


test_functions = {
    "weather": Function(
        "weather",
        "Gets the weather for a given area",
        [
            Argument(
                name="location",
                type="string",
                description="City, state, province name or zip code",
                required=True,
            ),
        ],
        weather,
    ),
    "news": Function(
        "news",
        "Gets the latest news for a given topic",
        [
            Argument(
                name="topic",
                type="string",
                description="The topic of the news",
                required=False,
            ),
        ],
        news,
    ),
}


class TestAgent(Agent):
    def __init__(self, llm: LLM, functions: Dict[str, Function] = {}):
        super().__init__(llm, functions=functions)

    def add_function_output_to_prompt(
        self,
        prompt: str,
        function_calls: FunctionCalls,
    ) -> Union[str, Dict[str, str]]:
        for function_call in function_calls:
            function_name, arguments, output = function_call
            prompt = prompt.replace("{" + function_name + "}", output)

        return prompt

    def parse_response(self, response: str) -> Any:
        return response

    def __call__(self, location: str):
        prompt = "Weather: {weather} | News: {news} \nGive me information about {location}."
        prompt = prompt.replace("{location}", location)
        result = self.llm_call(prompt)
        print(result)
        return result


agent = TestAgent(gpt, test_functions)
prompt = (
    "Weather: {weather} | News: {news} \nGive me information about San Diego"
)
response = agent(prompt)
print(response)
