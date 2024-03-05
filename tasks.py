from crewai import Task
from textwrap import dedent


# This is an example of how to define custom tasks.
# You can define as many tasks as you want.
# You can also define custom agents in agents.py
class CustomTasks:
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"

    def research_task(self, agent):
        return Task(
            description=dedent(
                "Identify the next big trend in {topic}."
                "Focus on identifying pros and cons and the overall narrative."
                "Your final report should clearly articulate the key points"
                "its market opportunities, and potential risks."
                f"{self.__tip_section()}"
            ),
            expected_output="A comprehensive 3 paragraphs long report on the latest AI trends.",
            agent=agent,
        )

    def write_task(self, agent):
        return Task(
            description=dedent(
                "Compose an insightful article on {topic}."
                "Focus on the latest trends and how it's impacting the industry."
                "This article should be easy to understand, engaging, and positive."
                f"{self.__tip_section()}"
            ),
            expected_output="A 4 paragraph article on {topic} advancements formatted as markdown.",
            agent=agent,
            async_execution=False,
            output_file="new-blog-post.md",
        )
