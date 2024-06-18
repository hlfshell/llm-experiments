import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import openai
from openai.types.chat.chat_completion import ChatCompletion

from llms.agent import LLM
from llms.tools import Function


class OpenAIGPT(LLM):
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
    ) -> Tuple[str, Dict[str, Dict[str, Any]]]:
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

        function_calls = self.__parse_function_calls(response)

        return response.choices[0].message.content, function_calls

    def __parse_function_calls(
        self, response: ChatCompletion
    ) -> Dict[str, Dict[str, Any]]:
        if response.choices[0].message.tool_calls is None:
            return {}

        function_calls: Dict[str, Dict[str, Any]] = {}

        for tool_msg in response.choices[0].message.tool_calls:

            params = json.loads(tool_msg.function.arguments)

            function_calls[tool_msg.function.name] = params

        return function_calls
