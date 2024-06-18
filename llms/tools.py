from __future__ import annotations
from typing import Any, Callable, Dict, List, Union

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from abc import ABC, abstractmethod


class Parser(ABC):

    def __init__() -> None:
        pass

    @abstractmethod
    def generate_descriptor(func: Function) -> str:
        pass

    @abstractmethod
    def parse_arguments(arguments: Any) -> Dict[str, Any]:
        pass


class Result:
    def __init__(self, result: str):
        self.result = result

    def __str__(self):
        pass


class Argument:
    def __init__(self, name: str, description: str, type: str, required: bool = False):
        self.name = name
        self.description = description
        self.type = type
        self.required = required


class Function:
    def __init__(
        self,
        name: str,
        description: str,
        args: List[Argument],
        func: Callable,
    ):
        self.name = name
        self.description = description
        self.args = args
        self.func = func

    def __call__(self, args: Union[dict, ChatCompletionMessageToolCall]):
        if isinstance(args, ChatCompletionMessageToolCall):
            args = self.parse_openai_tool_call(args)
        return self.func(args)

    def __str__(self):
        args = ", ".join([f"{k}" for k, _ in self.args.items()])
        return f"{self.name}({args}): {self.description}"

    def to_openai(self) -> Dict:
        properties = {}
        required_args = []
        for arg in self.args:
            properties[arg.name] = {
                "type": arg.type,
                "description": arg.description,
            }
            if arg.required:
                required_args.append(arg.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_args,
                },
            },
        }
