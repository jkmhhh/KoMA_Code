import os
import textwrap
import time
from rich import print
from typing import List

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI, ChatOllama
from langchain.callbacks import get_openai_callback
from scenario.envScenario import EnvScenario
from langchain.schema import AIMessage, HumanMessage, SystemMessage


delimiter = "####"

example_message = textwrap.dedent(f"""\
        {delimiter} Driving scenario description:
        You are driving on a highway. Please note that vehicles will be merging from on-ramps into the far right lane, which may pose a safety risk to you. You need to avoid colliding with merging vehicles.
        You are driving on a road with 3 lanes, and you are currently driving in the second lane from the left. Your current position is `(254.29, 4.00)`, speed is 16.35 m/s, acceleration is -2.29 m/s^2, and lane position is 24.29 m.
        There are other vehicles driving around you, and below is their basic information:
        - Vehicle `672` will turn left when he find a safe distance in a few seconds. - Vehicle `672` is driving on the lane to your right and is parallel to you. The centre position of it is `(254.97, 8.00)`, length is 5.0 meters, speed is 24.61 m/s, acceleration is 0.67 m/s^2, and lane position is 24.97 m.
        - Vehicle `800` is driving on the lane to your left and is parallel to you. The centre position of it is `(255.65, 0.00)`, length is 5.0 meters, speed is 21.11 m/s, acceleration is -0.73 m/s^2, and lane position is 25.65 m.
        - Vehicle `288` is driving on the same lane as you and is ahead of you. The centre position of it is `(281.89, 4.00)`, length is 5.0 meters, speed is 22.01 m/s, acceleration is -1.41 m/s^2, and lane position is 51.89 m.
        {delimiter} Your available actions:
        Turn-left - change lane to the left of the current lane Action_id: 0
        IDLE - remain in the current lane with current speed Action_id: 1
        Turn-right - change lane to the right of the current lane Action_id: 2
        Acceleration - accelerate the vehicle and increase the speed by 5m/s Action_id: 3
        Deceleration - decelerate the vehicle and reduce the speed by 5m/s Action_id: 4
        """)
example_answer = textwrap.dedent(f"""\
        Analyzing the intentions of other vehicles:
        - Vehicle 672 is planning to turn left soon, which means it might attempt to merge into the lane to its left, currently occupied by us. However, given its significantly higher speed (24.61 m/s) compared to ours (16.35 m/s), it is unlikely to slow down and merge behind us without significant deceleration.
        - Vehicle 800, on our left, is moving at a speed of 21.11 m/s but is decelerating. This vehicle does not pose an immediate threat to our lane change plan but could affect decisions regarding speed adjustments.
        - Vehicle 288, in the same lane as us and ahead, is moving faster than us (22.01 m/s) and is also decelerating. This vehicle's behavior does not directly impact our immediate decision-making but is important for maintaining safe following distances.

        Previous plan evaluation:
        The previous plan was to decelerate to allow Vehicle 672 to safely merge in front of us if it chooses to do so. Given Vehicle 672's current speed and position, it is unlikely to merge behind us without significant deceleration, which makes our previous plan of decelerating less relevant. Our current speed has already been reduced to 16.35 m/s due to previous deceleration, which is within the safe speed range but on the lower end. Further deceleration might not be necessary or beneficial at this point, especially considering that maintaining a stable speed could be more predictable for other drivers around us.

        Considering the current scenario and the intentions of the surrounding vehicles, the most reasonable and safe action seems to be to maintain our current lane and speed. This decision is based on the following:
        - Vehicle 672's high speed makes it unlikely to merge behind us safely without significant speed reduction.
        - Our current speed is already relatively low, and further deceleration could bring us closer to the lower limit of the safe speed range, potentially disrupting the flow of traffic.
        - Maintaining our current lane and speed (IDLE) provides predictability for other drivers, which is crucial for safety in dense traffic situations.

        Response to user:
        #### Plan B: Decelerate to allow Vehicle 672 to safely merge in front of us if it chooses to do so.
        #### 1""")


class DriverAgent:
    def __init__(
        self, sce: EnvScenario,
        temperature: float = 0, verbose: bool = False
    ) -> None:
        self.sce = sce
        oai_api_type = os.getenv("OPENAI_API_TYPE")
        if oai_api_type == "azure":
            print("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
                deployment_name="GPT-16",
                temperature=temperature,
                max_tokens=2000,
                request_timeout=60,
            )
        elif oai_api_type == "openai":
            print("Using Openai Chat API")
            self.llm = ChatOpenAI(
                temperature=temperature,
                model_name= 'gpt-4-turbo-preview', #'gpt-4',  # or any other model with 8k+ context gpt-4-32k gpt-3.5-turbo-16k
                max_tokens=2000,
                request_timeout=60,
            )
        elif oai_api_type == "ollama":
            print("Using ollama")
            self.llm = ChatOllama(model="llama3") # or any other model in ollama
    def few_shot_decision(self, scenario_description: str = "Not available", available_actions: str = "Not available", driving_attentions: str = "Not available", fewshot_messages: List[str] = None, fewshot_answers: List[str] = None, previous_plan: str = "Not available"):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/

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

        human_message = f"""\
        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario.
        
        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Driving attentions:
        {driving_attentions}
        {delimiter} Available actions:
        {available_actions}
        {delimiter} Previous plan:
        {previous_plan}
        
        You need to make decisions according to the following process:
        1.Try to analyze the intentions of other vehicles by their status information.
        2.Check whether there is already a previously established plan. 
            If there is no plan, you need to develop a plan to guide subsequent action. The following is the process for generating the plan:
                Step 1:Brainstorm all workable and distinct plans based on the current scenario. Here are some example of the plan's content: "decelerate and then merge behind Vehicle '87'" , "merge ahead of Vehicle '87'".
                Step 2:For each of the proposed plans, evaluate their potential. Consider their pros and cons, implementation difficulty, potential challenges. Assign safety, efficiency score from 0 to 10 to each option based on these factors. 
                Step 3:Based on the evaluations and scenarios, rank the plans.
                Step 4:Choose one plan as your driving plan according to your own idea.
            If you already have a previous plan, you need to firstly check whether the plan was finished. Then analyze the intentions of other vehicles to deduce whether the previous plan is still reasonable in the current situation. Output your reasoning process.Normally, your speed should not exceed 30m/s.
        3.If the previous plan is not workable or or has already been completed, then generate a new plan by using the steps described above. Otherwise, just keep the previous plan as the current plan.
        4.Analyse all the available actions and then make reasonable action choices based on the current scene information and the plan. If the plan was to slow down or speed up first and then change lanes, you need to firstly analyze whether you can make the lane change to complete the plan now. The most important thing is that the next decision is 0.5s later, so you need to ensure that the state after the decision is executed for 0.5s is safe. If the distance to the end of the ramp is less than your speed multiplied by the decision interval time and you do not choose change lanes, you will have a collision with the end of the ramp.
        5.Attentions: Each step needs you to reasoning. Changing lanes into the lane of a vehicle which parallel to you can cause a collision! Do not change lanes left and right frequently.
        You can stop reasoning once you find the best action to take. 
        """
        human_message = human_message.replace("        ", "")

        if fewshot_messages is None:
            raise ValueError("fewshot_message is None")
        messages = [
            SystemMessage(content=system_message),#system_message
            HumanMessage(content=example_message),#example_message
            AIMessage(content=example_answer),#example_answer
        ]

        for i in range(len(fewshot_messages)):
            messages.append(
                HumanMessage(content=fewshot_messages[i])
            )
            messages.append(
                AIMessage(content=fewshot_answers[i])
            )
        messages.append(
            HumanMessage(content=human_message)
        )
        start_time = time.time()
        response = self.llm(messages)
        print("Time used: ", time.time() - start_time)
        print(response.content)
        new_plan = response.content.split(delimiter)[-2]
        decision_action = response.content.split(delimiter)[-1]
        try:
            result = int(decision_action)
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            print("Output is not a int number, checking the output...")
            check_message = f"""
            You are a output checking assistant who is responsible for checking the output of another agent.
            
            The output you received is: {decision_action}

            Your should just output the right int type of action_id, with no other characters or delimiters.
            i.e. :
            | Action_id | Action Description                                     |
            |--------|--------------------------------------------------------|
            | 0      | Turn-left: change lane to the left of the current lane |
            | 1      | IDLE: remain in the current lane with current speed   |
            | 2      | Turn-right: change lane to the right of the current lane|
            | 3      | Acceleration: accelerate the vehicle                 |
            | 4      | Deceleration: decelerate the vehicle                 |


            You answer format would be:
            {delimiter} <correct action_id within 0-4>
            """
            messages = [
                HumanMessage(content=check_message),
            ]
            with get_openai_callback() as cb:
                check_response = self.llm(messages)
            result = int(check_response.content.split(delimiter)[-1])

        few_shot_answers_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_answers_store += fewshot_answers[i] + \
                "\n---------------\n"
        return result, response.content, human_message, few_shot_answers_store, new_plan
