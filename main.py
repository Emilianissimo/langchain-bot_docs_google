import sys

from langchain.tools import Tool

from constants import GREEN, WHITE
from factories.llm_factory import LLMFactory
from langchain.agents import initialize_agent, AgentType
from helpers.tool_error_handler import tool_error_handler


# Init all required components
llm: LLMFactory = LLMFactory()


# Setup tools
tools = [
    Tool.from_function(
        func=llm.chain.run,
        name='Document-parser-and-conversation-tool',
        description='useful to search data in the documents',
    ),
    Tool.from_function(
        func=llm.google_search_engine.run,
        name='Google-Custom-Search-Engine',
        description="useful to search data in the google if the data is not in the documents",
        handle_tool_error=tool_error_handler
    ),
]


agent = initialize_agent(
    tools,
    llm.llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,
    memory=llm.memory,
    handle_parsing_errors=True,
)


while True:
    query: str = input(f'{GREEN}Prompt (write exit to end or press F): ')

    if query in ['exit', 'F', 'f']:
        print('Finishing')
        sys.exit()

    if query == '':
        continue

    result = agent.run(query)
    print(f"{WHITE}Answer: " + result)

