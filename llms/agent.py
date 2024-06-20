from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional, Tuple, Union

from llms.tools import Function

# A RolePrompt is a dict specifying a role, and a string specifying the
# content. An example of this would be:
# { "role": "system", "content": "You are a an assistant AI whom should answer
# all questions in a straightforward manner" }
# { "role": "user", "content": "How much wood could a woodchuck chuck..." }
RolePrompt = Dict[str, str]

# Prompt is a union type - either a straight string, or a RolePrompt.
Prompt = Union[str, RolePrompt]

# FunctionArguments are a dict of the arguments passed to a function, with the
# key being the argument name and the value being the argument value.
FunctionArguments = Dict[str, Any]

# FunctionCalls are a type representing a set of of possible function calls and
# their results, representing a history of queries from an LLM to their
# functions. The format is a list of tuples; each tuple represents the name of
# the function, a FunctionArguments, and finally the return result of that
# function (type dependent on the function).
FunctionCalls = List[Tuple[str, FunctionArguments, Any]]


class LLM(ABC):
    def __init__(self, max_tokens: int = 512, temperature: float = 0.7):
        self.max_tokens = max_tokens
        self.temperature = temperature
        super().__init__()

    @abstractmethod
    def generate(
        self,
        prompt: Prompt,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: List[Function] = [],
    ) -> Tuple[str, Dict[str, FunctionArguments]]:
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
        clear_function_on_iteration: bool = False,
    ):
        self._llm = llm
        self._functions = functions

        self.__thread_pool = ThreadPoolExecutor(max_function_threads)
        self.__default_function_timeout = default_function_timeout

        self.__clear_function_on_iteration = clear_function_on_iteration

        super().__init__()

    @abstractmethod
    def parse_response(self, response: str) -> Any:
        """
        Given a response from the llm, provide any additional parsing required
        to give it back in the expected form if required. Note that it can be
        any return type - whatever your agent is expected to produce on a call.

        This is called after the LLM generates a response, but does not need
        any additional processing (such as from functions).
        """
        pass

    @abstractmethod
    def add_function_output_to_prompt(
        self, prompt: Prompt, function_output: FunctionCalls
    ) -> Prompt:
        """
        Given the *original* prompt and the resulting outputs of all function
        call results, add the function's results back into the prompt.

        How this is done is up to the implementing class. It can be
        incorporated directly into a prompt via templating or some other
        manner.

        The return is a new prompt with the function call data implemented,
        allowing the agent to continue work on the original request.
        """
        pass

    def call_functions(
        self, function_calls: Dict[str, FunctionArguments]
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

    def llm_call(
        self,
        prompt: Prompt,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        max_depth: Optional[int] = None,
        function_calls: FunctionCalls = [],
    ) -> Any:
        """
        llm_call handles the calling of the internal LLM.

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
        the `function_calls` populated. The `function_calls` is a list of
        function outputs in Tuple form. Each Tuple is of the format str (the
        function name), the parameters passed to that function, and the
        resulting output of that function, whatever it may be. If the max depth
        is surpassed doing this cycle, we will raise a MaxDepthError.

        When function_calls is populated, add_function_output_to_prompt is
        called to add the function outputs into the prompt in whatever manner
        is deemed best by the implementing class.

        This is then passed to the agent in some manner based on what the agent
        expects. If the setting clear_function_on_iteration is set to True,
        then the function_calls will be cleared between each call of the agent;
        otherwise each consecutive call will still be passed all of the
        function outputs from the prior calls.

        This function is recursively called until no more function calls are
        requested, and the resulting output is passed into parse_response.  The
        output of this function is finally returned as the result.
        """
        if max_depth is not None and max_depth <= 0:
            raise MaxDepthError(max_depth)

        if len(function_calls) > 0:
            expanded_prompt = self.add_function_output_to_prompt(
                prompt, function_calls
            )

            response, function_arguments = self._llm.generate(
                expanded_prompt, max_tokens, temperature, self._functions
            )
        else:
            response, function_arguments = self._llm.generate(
                prompt, max_tokens, temperature, self._functions
            )

        if function_arguments is None or len(function_arguments) == 0:
            return self.parse_response(response)
        else:
            function_results = self.call_functions(function_arguments)

            # Get our function results into the proper format
            new_function_calls = []
            for function_name, result in function_results.items():
                function_args = function_arguments[function_name]
                new_function_calls.append(
                    (function_name, function_args, result)
                )

            if self.__clear_function_on_iteration:
                function_calls = new_function_calls
            else:
                function_calls += new_function_calls

            return self.llm_call(
                prompt,
                max_tokens,
                temperature,
                max_depth - 1 if max_depth else None,
                function_calls,
            )

    @abstractmethod
    def __call__() -> Any:
        """
        __call__ is the main entry point for the agent. It should handle the
        calling of the LLM, the parsing of the response, and any other
        necessary steps to return the output in the expected format.

        We expect the agent to, at some point, call llm_call from the resulting
        input to utilize the agent function utilization and LLM management.
        """
        pass


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
    ) -> Tuple[Prompt, Dict[str, Dict[str, Any]]]:
        """
        parse_response will take the response from the agent at the given index
        and expects to be parsed and prepared for the next agent. It should
        return the response and any function calls that need to be made in the
        next agent.
        """
        pass

    @abstractmethod
    def parse_intermediate_response(
        self, index: int, response: Any
    ) -> Tuple[Union[str, dict], Dict[str, Dict[str, Any]]]:
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
