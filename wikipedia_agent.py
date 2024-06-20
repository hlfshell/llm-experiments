from agents.agent import LLM, Agent, FunctionCalls, Prompt
from agents.functions.wikipedia import WikipediaPage, WikipediaTopicQuery
from agents.llms.openai import OpenAIGPT
from agents.templater import PromptTemplate


class WikipediaAgent(Agent):
    def __init__(self, llm: LLM):
        self.prompt = PromptTemplate.Load(
            "./prompts/simple_wikipedia_agent.prompt"
        )
        super().__init__(
            llm,
            functions={
                "wikipedia_page": WikipediaPage(),
                "wikipedia_topic_query": WikipediaTopicQuery(),
            },
        )

    def parse_response(self, response: str) -> str:
        return response

    def add_function_output_to_prompt(
        self, prompt: Prompt, function_output: FunctionCalls
    ) -> Prompt:
        pass


wiki = WikipediaAgent(OpenAIGPT())
