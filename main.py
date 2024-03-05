import os
from crewai import Crew, Process
from decouple import config
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_fireworks import ChatFireworks

from textwrap import dedent
from agents import CustomAgents
from tasks import CustomTasks


search_tool = DuckDuckGoSearchRun()

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = config("SERPER_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = config("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = config("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = config("LANGCHAIN_PROJECT")
os.environ["FIREWORKS_API_KEY"] = config("FIREWORKS_API_KEY")

# This is the main class that you will use to define your custom crew.
# You can define as many agents and tasks as you want in agents.py and tasks.py


class CustomCrew:
    def __init__(self, user_input):
        self.user_input = user_input
        self.mixtral_8x7b = ChatFireworks(
            model_name="accounts/fireworks/models/mixtral-8x7b-instruct",
            temperature=0.7,
        )
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = CustomAgents()
        tasks = CustomTasks()

        # Define your custom agents and tasks here
        researcher = agents.researcher()
        writer = agents.writer()

        # Custom tasks include agent name and variables as input
        research_task = tasks.research_task(
            researcher,
        )

        write_task = tasks.write_task(
            writer,
        )

        # Define your custom crew here
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            process=Process.hierarchical,
            manager_llm=self.OpenAIGPT35,
            verbose=True,
        )

        result = crew.kickoff(inputs={"topic": self.user_input})
        return result


# This is the main function that you will use to run your custom crew.
if __name__ == "__main__":
    print("## Welcome to Crew AI")
    print("-------------------------------")
    user_input = input(dedent("""User's input: """))

    custom_crew = CustomCrew(user_input)
    result = custom_crew.run()
    print("\n\n########################")
    print("## Here is you custom crew run result:")
    print("########################\n")
    print(result)
