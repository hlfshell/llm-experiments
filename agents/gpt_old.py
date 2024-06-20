import json
import os
from typing import Dict, List, Optional, Union

import openai
from openai.types.chat.chat_completion_message_tool_call import \
    ChatCompletionMessageToolCall

from llms.tools import Argument, Function


class GPT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        engine: str = "3.5",
        default_temperature: float = 0.7,
        tools: Dict[str, Function] = {},
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key is None:
            raise ValueError("API Key is required")

        if engine == "3.5":
            self.engine = "gpt-3.5-turbo"
        elif engine == "4":
            self.engine = "gpt-4"
        else:
            self.engine = engine

        self.default_temperature = default_temperature

        self.tools = tools

        self.client = openai.Client(api_key=self.api_key)

    def parse_openai_tool_call(self, data: ChatCompletionMessageToolCall) -> dict:
        return json.loads(data.function.arguments)

    def generate(
        self,
        prompt: Union[str, dict],
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ) -> str:
        if temperature is None:
            temperature = self.default_temperature

        if isinstance(prompt, str):
            prompt = [{"role": "system", "content": prompt}]

        tools = [tool.to_openai() for tool in self.tools.values()]

        response = self.client.chat.completions.create(
            model=self.engine,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
        )

        print(response)
        print(type(response.choices[0].message.tool_calls[0]))
        print("tools")

        # Determine if any tools were called
        for tool_msg in response.choices[0].message.tool_calls:
            print(tool_msg)

            # Build params
            # params = Argument.FromOpenAI(tool.parameters)
            # pass
            print("***")
            print(self.parse_openai_tool_call(tool_msg))
            print("***")
            params = self.parse_openai_tool_call(tool_msg)

            # identify the tool this call refers to
            # amongst tools we have
            if tool_msg.function.name not in self.tools:
                raise "unidentified tool"
            else:
                tool = self.tools[tool_msg.function.name]

            # Call function
            results = tool(params)
            print(results)

            return
