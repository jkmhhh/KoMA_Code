import os
import textwrap
import time
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from rich import print


class ReflectionAgent:
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

    def reflection(self, human_message: str, llm_response: str, efficiency_score:int, safety_score:int, is_collision:bool) -> str:
        delimiter = "####"
        system_message = textwrap.dedent(f"""\
                You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
                You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.

                Your response should use the following format:
                <reasoning>
                <reasoning>
                <repeat until you have a decision>
                Response to user:
                {delimiter} <output the plan you used>
                {delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 
                Make sure to include {delimiter} to separate every step.
                """)
        human_message = textwrap.dedent(f"""\
            ``` Human Message ```
            {human_message}
            ``` ChatGPT Response ```
            {llm_response}
            ``` Past Action Score ```
            Efficiency_score:{efficiency_score} Safety_score:{safety_score} is_collision:{is_collision}
            
            Now, you know this action ChatGPT output cause a collision or result a low score after executing this action, which means there are some mistake in ChatGPT resoning and cause the wrong plan or action.    
            Please carefully check every reasoning in ChatGPT response and find out the mistake in the reasoning process of ChatGPT, and also output your corrected version of ChatGPT response.
            Your answer should use the following format:
            {delimiter} Analysis of the mistake:
            <Your analysis of the mistake in ChatGPT reasoning process>
            {delimiter} What should ChatGPT do to avoid such errors in the future:
            <Your answer>
            {delimiter} Corrected version of ChatGPT response:
            <Your corrected version of ChatGPT response>
        """)

        print("Reflection is running ...")
        start_time = time.time()
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]
        response = self.llm(messages)
        target_phrase = f"{delimiter} What should ChatGPT do to avoid such errors in the future:"
        substring = response.content[response.content.find(
            target_phrase)+len(target_phrase):].strip()
        corrected_memory = f"{delimiter} I have made a misake before and below is my self-reflection:\n{substring}"
        print("Reflection done. Time taken: {:.2f}s".format(
            time.time() - start_time))
        print("corrected_memory:", corrected_memory)

        return corrected_memory
