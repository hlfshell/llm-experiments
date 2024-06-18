import os
from typing import Any, Dict, List, Optional, Union

import openai

from llms.agent import LLM, Agent
from llms.tools import Argument, Function


class GPT(LLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        engine: str = "3.5",
        default_temperature: float = 0.7,
    ):
        super().__init__(max_tokens=512, temperature=default_temperature)

        if engine == "3.5":
            self.engine = "gpt-3.5-turbo"
        elif engine == "4":
            self.engine = "gpt-4"
        else:
            self.engine = engine

        self.__api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.client = openai.Client(api_key=self.__api_key)

    def __function_to_tool(self, func: Function) -> Dict:
        properties = {}
        required_args = []
        for arg in func.args:
            properties[arg.name] = {
                "type": arg.type,
                "description": arg.description,
            }
            if arg.required:
                required_args.append(arg.name)

        return {
            "type": "function",
            "function": {
                "name": func.name,
                "description": func.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_args,
                },
            },
        }

    def generate(
        self,
        prompt: Union[str, List[dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Dict[str, Function] = {},
    ) -> str:
        if temperature is None:
            temperature = self.temperature

        if isinstance(prompt, str):
            prompt = [{"role": "system", "content": prompt}]

        functions_instruct = [
            self.__function_to_tool(function) for function in functions.values()
        ]
        if len(functions_instruct) == 0:
            functions_instruct = None

        response = self.client.chat.completions.create(
            model=self.engine,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=functions_instruct,
        )

        return response.choices[0].message.content

    def parse_function_calls(self, Any) -> Dict[str, Dict[str, Any]]:
        pass


class TestAgent(Agent):
    def __init__(self, llm: LLM):
        super().__init__(llm, functions={})

    def call_functions(
        self, function_calls: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        pass

    def add_function_output_to_prompt(
        self, prompt: str, function_output: Dict[str, Any]
    ) -> Union[str, Dict[str, str]]:
        pass

    def parse_response(self, response: str) -> Any:
        pass


agent = TestAgent(GPT())
agent("Hello world")
