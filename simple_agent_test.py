import os
from typing import Any, Dict, List, Optional, Union

import openai

from llms.agent import LLM, Agent
from llms.openai import OpenAIGPT
from llms.tools import Argument, Function

gpt = OpenAIGPT()


def crasher(*args):
    print("Crasher:", args)
    raise "crash"


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
        crasher,
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
        crasher,
    ),
}


class TestAgent(Agent):
    def __init__(self, llm: LLM, functions: Dict[str, Function] = {}):
        super().__init__(llm, functions=functions)

    def add_function_output_to_prompt(
        self, prompt: str, function_output: Dict[str, Any]
    ) -> Union[str, Dict[str, str]]:
        pass

    def parse_response(self, response: str) -> Any:
        return response


agent = TestAgent(gpt, test_functions)
response = agent("Hello world")
print(response)
