from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional, Tuple, Union

from llms.tools import Function


class LLM(ABC):
    def __init__(self, max_tokens: int = 512, temperature: float = 0.7):
        self.max_tokens = max_tokens
        self.temperature = temperature
        super().__init__()

    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: List[Function] = [],
    ) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """
        Generate a response from the LLM given a prompt. It returns a tuple -
        first the string output of the generated prompt (if any), and a set of
        function calls from the LLM.

        The prompt can be a str or a list of dicts, both of which are common for
        LLM applications. It is up to the implementer to properly deal/convert
        these.

        max_tokens and temperature are possible overrides for the default and
        must be handled. If max_tokens or temperature is None fall back to the
        desired sensible defaults.

        If functions are to be used on this call, they are passed in as a list
        of functions - it is up to the implemented LLM class to incorporate
        these for the target LLM (templating into the prompt, specific
        instruction, etc).
        """
        pass


class Agent(ABC):

    def __init__(
        self,
        llm: LLM,
        functions: Dict[str, Function],
        max_function_threads: int = 5,
        default_function_timeout: int = 60,
    ):
        self._llm = llm
        self._functions = functions

        self.__thread_pool = ThreadPoolExecutor(max_function_threads)
        self.__default_function_timeout = default_function_timeout

        super().__init__()

    @abstractmethod
    def parse_response(self, str) -> Any:
        """
        Given a response from the llm, provide any additional parsing required
        to give it back in the expected form if required.
        """
        pass

    @abstractmethod
    def add_function_output_to_prompt(
        self, prompt: str, function_output: Dict[str, Any]
    ) -> Union[str, Dict[str, str]]:
        """
        Given a prompt and the resulting outputs of a function
        call (in the format of a dict, where the key is the
        name of the function and the return is whatever one
        would expect from the function), add the function's
        results back into the prompt.

        How this is done is up to the implementing class. It
        can be incorporated directly into a prompt via
        templating or some other manner.
        """
        pass

    def call_functions(
        self, function_calls: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Given a dict of function calls, call each function and return a dict of
        function name and result.

        If the number of function calls is greater than the max function
        threads, then we should handle this gracefully.
        """
        results: Dict[str, Any] = {}

        futures: Dict[str, Future] = {}
        for function_name, args in function_calls.items():
            if function_name not in self._functions:
                raise FunctionNotFoundError(function_name)

            futures[function_name] = self.__thread_pool.submit(
                self._functions[function_name], args
            )
        wait(futures.values(), timeout=self.__default_function_timeout)

        for function_name in futures.keys():
            future = futures[function_name]
            if future.exception():
                raise future.exception()

            result = future.result()
            results[function_name] = result

        return results

    def __call__(
        self,
        prompt: Union[str, dict],
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        max_depth: Optional[int] = None,
        function_outputs: List[Tuple[str, Dict[str, Any], Any]] = [],
    ) -> Any:
        """
        Respond triggers the incoming prompt.

        We assume that the prompt variable is a set of dicts and, if it's a
        string we convert to:

        { "role": "system", "content": prompt }

        The max depth, if not set, will allow the functions to be called
        endlessly.

        When respond is called, we will check for any calls to available
        functions. If function calls are within the generated response (however
        that may be handled), then we will call each function, compile the
        generated output for each function, and then call the LLM again with
        the `function_outputs` populated. The `function_outputs` is a list of
        function outputs in Tuple form. Each Tuple is of the format str (the
        function name), the parameters passed to that function, and the
        resulting output of that function, whatever it may be. If the max depth
        is surpassed doing this cycle, we will raise a MaxDepthError.

        This is then passed to the agent in some manner based on what the agent
        expects.

        This function is recursively called until no more function calls are
        requested, and the resulting output is passed back.
        """
        if max_depth is not None and max_depth <= 0:
            raise MaxDepthError(max_depth)

        response, function_calls = self._llm.generate(
            prompt, max_tokens, temperature, self._functions, function_outputs
        )
        print("!!!", response, function_calls)

        if function_calls is None or len(function_calls) == 0:
            return self.parse_response(response)
        else:
            function_results = self.call_functions(function_calls)

            new_prompt = self._llm.add_function_output_to_prompt(
                response, function_results
            )

            return self.respond(
                new_prompt,
                max_tokens,
                temperature,
                max_depth - 1 if max_depth else None,
            )


class SequentialAgent(ABC):
    """
    SequentialAgent will sequentially pass the prompt to a series of agents,
    feeding responses into the next agent.

    You are expected to implement this class's
    """

    def __init__(
        self,
        agents: List[Agent],
    ):
        self.agents = agents

    @abstractmethod
    def parse_response(
        self, index: int, response: Any
    ) -> Tuple[Union[str, dict], Dict[str, Dict[str, Any]]]:
        """
        parse_response will take the response from the agent at the given index
        and expects to be parsed and prepared for the next agent. It should
        return the response and any function calls that need to be made in the
        next agent.
        """
        pass

    def __call__(
        self,
        prompt: Union[str, dict],
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        max_depth: Optional[int] = None,
        function_outputs: List[Tuple[str, Dict[str, Any], Any]] = [],
    ) -> Any:

        for agent in self.agents:
            response = agent(
                prompt, max_tokens, temperature, max_depth, function_outputs
            )
            prompt, function_outputs = self.parse_response(response)


class ParallelAgent(ABC):
    """
    ParallelAgent will pass the prompt to a series of agents in parallel, and
    then return each agent's response.

    If a dict of agents is passed, the response will be a dict of agent names
    and their resulting outputs. If a list is passed, the response will be a
    list of outputs in the order the agents were passed.
    """

    def __init__(
        self,
        agents: Union[Dict[str, Agent], List[Agent]],
        max_workers: int = 5,
        timeout: float = 60.0,
    ):
        if len(agents) == 0:
            raise ValueError("Agents must be provided")

        self.__type = "list" if isinstance(agents, list) else "dict"

        self.agents = agents
        self.__thread_pool = ThreadPoolExecutor(max_workers=len(agents))
        self.__timeout = timeout

    def __call__(
        self,
        prompt: Union[str, dict],
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        max_depth: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any]]:
        if self.__type == "list":
            responses = []

            futures: List[Future] = []
            for agent in self.agents:
                futures.append(
                    self.__thread_pool.submit(
                        agent,
                        prompt,
                        max_tokens,
                        temperature,
                        max_depth,
                    )
                )
            wait(futures, timeout=self.__timeout)

            responses = []
            for future in futures:
                if future.exception():
                    raise future.exception()
                responses.append(future.result())

        else:
            futures: Dict[str, Future] = {}
            for agent_name, agent in self.agents.items():
                futures[agent_name] = self.__thread_pool.submit(
                    agent,
                    prompt,
                    max_tokens,
                    temperature,
                    max_depth,
                )

            wait(futures.values(), timeout=self.__timeout)

            responses = {}
            for agent_name, future in futures.items():
                if future.exception():
                    raise future.exception()
                responses[agent_name] = future.result()

        return responses


class MaxDepthError(Exception):
    def __init__(self, depth: int) -> None:
        self.message = f"Max depth of {depth} reached"
        super().__init__(self.message)


class FunctionNotFoundError(Exception):
    def __init__(self, function_name: str) -> None:
        self.message = f"Function {function_name} is not a known function"
        super().__init__(self.message)
