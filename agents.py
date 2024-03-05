from crewai import Agent
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from langchain_fireworks import ChatFireworks


# This is an example of how to define custom agents.
# You can define as many agents as you want.
# You can also define custom tasks in tasks.py
class CustomAgents:
    def __init__(self):
        self.mixtral_8x7b = ChatFireworks(
            model_name="accounts/fireworks/models/mixtral-8x7b-instruct",
            temperature=0.7,
        )
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.search_tool = SerperDevTool()

    def researcher(self):
        return Agent(
            role="Senior Researcher",
            goal="Uncover groundbreaking technologies in {topic}",
            verbose=True,
            memory=True,
            backstory=(
                "Driven by curiosity, you're at the forefront of"
                "innovation, eager to explore and share knowledge that could change"
                "the world."
            ),
            tools=[self.search_tool],
            llm=self.OpenAIGPT35,
            allow_delegation=True,
        )

    def writer(self):
        return Agent(
            role="Writer",
            goal="Narrate compelling tech stories about {topic}",
            verbose=True,
            memory=True,
            backstory=(
                "With a flair for simplifying complex topics, you craft"
                "engaging narratives that captivate and educate, bringing new"
                "discoveries to light in an accessible manner."
            ),
            tools=[self.search_tool],
            llm=self.OpenAIGPT35,
            allow_delegation=False,
        )
