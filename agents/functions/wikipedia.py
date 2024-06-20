from typing import List

import wikipedia

from agents.tools import Argument, Function


class WikipediaTopicQuery(Function):

    def __init__(self):
        super().__init__(
            "wikipedia_topic_query",
            "Search Wikipedia for articles that match a given query topic",
            [
                Argument(
                    name="query",
                    type="string",
                    description="The query to search for",
                    required=True,
                ),
            ],
            self.topic_query,
        )

    def topic_query(self, topic: str) -> List[str]:
        wikipedia.search(topic)


class WikipediaPage(Function):

    def __init__(self):
        super().__init__(
            "wikipedia_page",
            "Get the content of a Wikipedia page",
            [
                Argument(
                    name="title",
                    type="string",
                    description="The title of the Wikipedia page - "
                    + "returns 'None' if the page does not exist",
                    required=True,
                ),
            ],
            self.page,
        )

    def page(self, title: str) -> str:
        return wikipedia.page(title).content
