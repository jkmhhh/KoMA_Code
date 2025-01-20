import copy
import gymnasium as gym
import numpy as np
import pandas as pd
import os
# base setting
from scenario.envScenario import EnvScenario
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.vectorStore import DrivingMemory
from LLMDriver.reflectionAgent import ReflectionAgent
from LLMDriver.reflection_choose_agent import Reflection_Choose_Agent
from gymnasium.wrappers import RecordVideo
from langchain.callbacks import get_openai_callback
from loadConfig import load_openai_config

USE_MEMORY = False
REFLECTION = False

encode_type = 'sce_language'
db_path = 'db/test'
result_folder = "./result/test"
few_shot_num = 2 # The number of memory fragments returned by the memory module
simulation_duration = 20 
load_openai_config()
# environment setting

config={
    'KoMA-merge-generalization':
        {
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "TimeToCollision",

                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "target_speeds":np.linspace(0,40,9),
                },
            },
            "simulation_frequency": 100,  # [Hz]
            "policy_frequency": 2,  # [Hz] The number of actions performed per second
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 1400,  # [px]
            "screen_height": 200,  # [px]
            "centering_position": [0.5, 0.5],
            "scaling": 5.5,
            "show_trajectories": True,
            "render_agent": False,
            "offscreen_rendering": False,
            "other_vehicles_count": 5,
            "controlled_vehicles_count": 2
        }
}
if USE_MEMORY:
    agentMemory = DrivingMemory(encode_type=encode_type, db_path=db_path)

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
with open(result_folder + "/" + 'log.txt', 'a') as f:
    f.write("result_folder {}\n".format(
        result_folder))
controlled_vehicle_number = config['KoMA-merge-generalization']["controlled_vehicles_count"]
episode = 0

while episode < simulation_duration:
    envType = 'KoMA-merge-generalization'
    env = gym.make(envType, render_mode='rgb_array')
    env.unwrapped.configure(config[envType])
    result_prefix = f"exp_{episode}"
    env = RecordVideo(env, result_folder, name_prefix=result_prefix)
    env.unwrapped.set_record_video_wrapper(env)
    obs, info = env.reset()
    env.render()
    sce = EnvScenario(env, envType)
    DA_list=[]
    DA1 = DriverAgent(sce, verbose=True)
    DA2 = DriverAgent(sce, verbose=True)
    DA_list.append(DA1)
    DA_list.append(DA2)

    if REFLECTION:
        RA = ReflectionAgent(verbose=True)
        RCA = Reflection_Choose_Agent(verbose=True)

    docs_list = [[] for i in range(controlled_vehicle_number)]
    efficiency_score_list = [[] for i in range(controlled_vehicle_number)]
    safety_score_list = [[] for i in range(controlled_vehicle_number)]
    collision_list = [0 for i in range(controlled_vehicle_number)]

    break_flag = False
    try:
        with get_openai_callback() as cb:
            already_decision_steps = 0
            previous_plan_list = [None, None]
            for j in range(0, simulation_duration):
                collision_frame = -1
                obs = np.array(obs, dtype=float)
                action_list=[]
                for i in range(controlled_vehicle_number):
                    docs = docs_list[i]
                    sce_descrip = sce.describe(i)
                    avail_action = sce.availableActionsDescription(i)
                    previous_plan = previous_plan_list[i]
                    print(sce_descrip)
                    fewshot_messages = []
                    fewshot_answers = []
                    fewshot_actions = []
                    if USE_MEMORY:
                        fewshot_results = agentMemory.retriveMemory(
                            sce, i, few_shot_num)
                        for fewshot_result in fewshot_results:
                            fewshot_messages.append(
                                fewshot_result["human_question"])
                            fewshot_answers.append(fewshot_result["LLM_response"])
                            fewshot_actions.append(fewshot_result["action"])
                            mode_action = max(
                                set(fewshot_actions), key=fewshot_actions.count)
                            mode_action_count = fewshot_actions.count(mode_action)
                    action, response, human_question, fewshot_answer, new_plan = DA_list[i].few_shot_decision(
                        scenario_description=sce_descrip, available_actions=avail_action,
                        fewshot_messages=fewshot_messages,
                        driving_attentions="Drive safely and avoid collisions.No matter how fast the two vehicles are, changing lanes onto the lane of a parallel vehicle will always result in a collision",
                        fewshot_answers=fewshot_answers,
                        previous_plan=previous_plan_list[i])
                    action_list.append(action)
                    previous_plan_list[i] = new_plan
                    docs.append({
                        "controlled_vehicle_id": i,
                        "simulation_time": j,
                        "sce_description": sce_descrip,
                        "human_question": human_question,
                        "response": response,
                        "plan": new_plan,
                        "action": action,
                        "sce": copy.deepcopy(sce)
                    })
                action = tuple(action_list)
                obs, reward, done, info, _ = env.step(action)
                speed, efficiency_score, safety_score = sce.evaluation(controlled_vehicle_number)
                for k in range(controlled_vehicle_number):
                    docs_list[k][-1]["efficiency_score"] = efficiency_score[k]
                    docs_list[k][-1]["safety_score"] = safety_score[k]
                    docs_list[k][-1]["speed"] = speed[k]
                    efficiency_score_list[k].append(efficiency_score[k])
                    safety_score_list[k].append(safety_score[k])
                already_decision_steps += 1
                env.render()
                env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()
                if done:
                    print("[red]Simulation done with running steps: [/red] ", j)
                    collision_frame = j
                    for i in range(controlled_vehicle_number):
                        collision_list[i] = env.controlled_vehicles[i].crashed
                    print(collision_list)
                    break
    finally:
        print("==========Simulation {} Done==========".format(episode))
        print(cb)
        print("Simulation done")

        for controlled_vehicle in range(controlled_vehicle_number):
            is_collision = collision_list[controlled_vehicle]
            docs = docs_list[controlled_vehicle]
            e_list = efficiency_score_list[controlled_vehicle]
            s_list = safety_score_list[controlled_vehicle]
            path = result_folder + "/" + f"Simulation_{episode}.csv".format(episode=episode)
            pd.DataFrame(docs).to_csv(path_or_buf=path, mode='a')
            if REFLECTION and is_collision:  # has collision
                i = collision_frame
                corrected_response = RA.reflection(
                    docs[i]["human_question"], docs[i]["response"], e_list[i], s_list[i], is_collision)
                agentMemory.addMemory(
                    docs[i]["sce_description"],
                    docs[i]["human_question"],
                    corrected_response,
                    docs[i]["plan"],
                    docs[i]["action"],
                    docs[i]["sce"],
                    comments="collision-mistake-correction"
                    )

            else:
                if REFLECTION:
                    wrong_action = int(RCA.reflection_choose(e_list, s_list))
                    if wrong_action == -1:
                        for i in range(0, len(docs)):
                            agentMemory.addMemory(
                                docs[i]["sce_description"],
                                docs[i]["human_question"],
                                docs[i]["response"],
                                docs[i]["plan"],
                                docs[i]["action"],
                                docs[i]["sce"],
                                comments="no-mistake-direct"
                            )
                    else:
                        corrected_response = RA.reflection(
                            docs[wrong_action]["human_question"], docs[wrong_action]["response"], e_list[wrong_action], s_list[wrong_action], False)
                        agentMemory.addMemory(
                            docs[wrong_action]["sce_description"],
                            docs[wrong_action]["human_question"],
                            corrected_response,
                            docs[wrong_action]["plan"],
                            docs[wrong_action]["action"],
                            docs[wrong_action]["sce"],
                            comments="mistake-correction"
                        )
        episode += 1
        env.close()
