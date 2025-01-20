import os
import textwrap
import time
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI, ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from rich import print


class Reflection_Choose_Agent:
    def __init__(
            self, temperature: float = 0.0, verbose: bool = False
    ) -> None:
        oai_api_type = os.getenv("OPENAI_API_TYPE")
        if oai_api_type == "azure":
            print("Using Azure Chat API")
            print("[red]Cautious: Using GPT4 now, may cost a lot of money![/red]")
            self.llm = AzureChatOpenAI(
                deployment_name="GPT-4",
                temperature=temperature,
                max_tokens=1000,
                request_timeout=60,
            )
        elif oai_api_type == "openai":
            self.llm = ChatOpenAI(
                temperature=temperature,
                model_name='gpt-4-turbo-preview',
                max_tokens=1000,
                request_timeout=60,
            )
        elif oai_api_type == "ollama":
            print("Using ollama")
            self.llm = ChatOllama(model="llama3") # or any other model in ollama
    def reflection_choose(self, efficiency_score_list: list, safety_score_list: list) -> str:
        delimiter = "####"
        system_message = textwrap.dedent(f"""\
                You are ChatGPT, a large language model trained by OpenAI. Now you act as a data analysing assistant, who can analyze data to find problems.
                You will be given some lists of scores for each second of a previous drive episode. The score ranges from 0 to 10.If the score is extremely low（below 5） or plummeting, it indicates that there is a problem with the action decision of that second.
                Your goal is to analyze the score lists to determine whether the action decisions have gone wrong and then return the list index of the first wrong action. 
                If you think there is nothing wrong, the list index of wrong action is '-1'. 
                Your response should use the following format:
                <reasoning>
                <reasoning>
                <repeat until you have finished>
                Response to user:
                {delimiter} <only output the index of wrong action as a int number of you decision, without any explanation. The output must be unique and not ambiguous> 
                Make sure to include {delimiter} to separate every step.
                """)
        human_message = textwrap.dedent(f"""\   
            Efficiency_score_list: {efficiency_score_list}
            safety_score_list: {safety_score_list}
            Now, you should analyze each of the score list and return the list index of the first wrong action.
            Your answer should use the following format:
            Analysis of each list:<Your analysis of each list>
            {delimiter} <The list index of wrong action>
        """)

        print("Reflection is running ...")
        start_time = time.time()
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]
        response = self.llm(messages)
        print("Reflection done. Time taken: {:.2f}s".format(
            time.time() - start_time))
        print("Analysis:", response.content)
        wrong_action = response.content.split(delimiter)[-1]
        return wrong_action





