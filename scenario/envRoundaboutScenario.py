from typing import List, Tuple, Optional, Union, Dict
from datetime import datetime
import math
import os

from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.road.lane import (
    StraightLane, CircularLane, SineLane, PolyLane, PolyLaneFixedWidth
)  # sqlite
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
import numpy as np

from scenario.envPlotter import ScePlotter
from scenario.DBBridge import DBBridge

ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

ACTIONS_DESCRIPTION = {
    0: 'Turn-left - change lane to the left of the current lane',
    1: 'IDLE - remain in the current lane with current speed',
    2: 'Turn-right - change lane to the right of the current lane',
    3: 'Acceleration - accelerate the vehicle and increase the speed by 5m/s',
    4: 'Deceleration - decelerate the vehicle and reduce the speed by 5m/s'
}


class EnvScenario:
    def __init__(
            self, env: AbstractEnv, envType: str
    ) -> None:
        self.env = env
        self.envType = envType
        self.ego = env.controlled_vehicles[0]
        self.theta1 = math.atan(3 / 17.5)
        self.theta2 = math.atan(2 / 2.5)
        self.radius1 = np.linalg.norm([3, 17.5])
        self.radius2 = np.linalg.norm([2, 2.5])

        self.road: Road = env.road
        self.network: RoadNetwork = self.road.network

        self.plotter = ScePlotter()


    def getSurrendVehicles(self, vehicles_count: int) -> List[IDMVehicle]:
        return self.road.close_vehicles_to(
            self.ego, self.env.PERCEPTION_DISTANCE,
            count=vehicles_count - 1, see_behind=True,
            sort='sorted'
        )

    def plotSce(self, fileName: str) -> None:
        SVs = self.getSurrendVehicles(10)
        self.plotter.plotSce(self.network, SVs, self.ego, fileName)

    def getUnitVector(self, radian: float) -> Tuple[float]:
        return (
            math.cos(radian), math.sin(radian)
        )

    def isInJunction(self, vehicle: Union[IDMVehicle, MDPVehicle]) -> float:
        if self.envType == 'intersection-v1':
            x, y = vehicle.position
            if -20 <= x <= 20 and -20 <= y <= 20:
                return True
            else:
                return False
        else:
            return False

    def getLanePosition(self, vehicle: Union[IDMVehicle, MDPVehicle]) -> float:
        currentLaneIdx = vehicle.lane_index
        currentLane = self.network.get_lane(currentLaneIdx)
        if not isinstance(currentLane, StraightLane):
            raise ValueError(
                "The vehicle is in a junction, can't get lane position"
            )
        else:
            currentLane = self.network.get_lane(vehicle.lane_index)
            return np.linalg.norm(vehicle.position - currentLane.start)

    def availableActionsDescription(self, controlled_vehicle_id) -> str:
        avaliableActionDescription = 'Your available actions are: \n'
        availableActions = self.env.get_available_actions()
        action_id=[]
        for action in availableActions:
            action_id.append(action[controlled_vehicle_id])
        action_id = list(set(action_id))
        for action in action_id:
            avaliableActionDescription += ACTIONS_DESCRIPTION[action] + ' Action_id: ' + str(action) + '\n'
        return avaliableActionDescription

    def processNormalLane(self, lidx: LaneIndex) -> str:
        sideLanes = self.network.all_side_lanes(lidx)
        numLanes = len(sideLanes)
        if lidx == ('ses', 'se', 0): #'k', 'b', 0
            background = "You're approaching the roundabout. Your goal is to safely pass the roundabout, please make sure there is no collision with other vehicles.\n"
        elif lidx == ('se', 'ex', 1): #'b', 'c', numLanes - 1
            background = f"You are driving on the roundabout.Your goal is to maintain a safe distance from adjacent vehicles. Please make sure there is no collision with other vehicles.\n"
        else:
            background = "You're driving on city roads. Your goal is to drive safely.\n"

        if numLanes == 1:
            description = "You are driving on a road with only one lane, you can't change lane. "
        else:
            egoLaneRank = lidx[2]
            if egoLaneRank == 0:
                description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the leftmost lane. "
            elif egoLaneRank == numLanes - 1:
                description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the rightmost lane. "
            else:
                laneRankDict = {
                    1: 'second',
                    2: 'third',
                    3: 'fourth'
                }
                description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the {laneRankDict[egoLaneRank]} lane from the left. "

        description += f"Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, speed is {self.ego.speed:.2f} m/s, acceleration is {self.ego.action['acceleration']:.2f} m/s^2.\n"
        return background + description

    def getSVRelativeState(self, sv: IDMVehicle) -> str:
        relativePosition = sv.position - self.ego.position
        egoUnitVector = self.getUnitVector(self.ego.heading)
        cosineValue = sum(
            [x * y for x, y in zip(relativePosition, egoUnitVector)]
        )
        if cosineValue >= 5:          #sv.LENGTH/2+self.ego.LENGTH/2
            return 'is ahead of you'
        elif cosineValue <= -5:
            return 'is behind you'
        else:
            return 'is parallel to you'

    def getVehDis(self, veh: IDMVehicle):
        posA = self.ego.position
        posB = veh.position
        distance = np.linalg.norm(posA - posB)
        return distance

    def getClosestSV(self, SVs: List[IDMVehicle]):
        if SVs:
            closestIdex = -1
            closestDis = 99999999
            for i, sv in enumerate(SVs):
                dis = self.getVehDis(sv)
                if dis < closestDis:
                    closestDis = dis
                    closestIdex = i
            return SVs[closestIdex]
        else:
            return None

    def processSingleLaneSVs(self, SingleLaneSVs: List[IDMVehicle]):
        if SingleLaneSVs:
            aheadSVs = []
            behindSVs = []
            for sv in SingleLaneSVs:
                RSStr = self.getSVRelativeState(sv)
                if RSStr == 'is ahead of you':
                    aheadSVs.append(sv)
                else:
                    behindSVs.append(sv)
            aheadClosestOne = self.getClosestSV(aheadSVs)
            behindClosestOne = self.getClosestSV(behindSVs)
            return aheadClosestOne, behindClosestOne
        else:
            return None, None

    def processSVsNormalLane(
            self, SVs: List[IDMVehicle], currentLaneIndex: LaneIndex
    ):
        classifiedSVs: Dict[str, List[IDMVehicle]] = {
            'current lane': [],
            'left lane': [],
            'right lane': [],
            'target lane': []
        }
        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        for sv in SVs:
            lidx = sv.lane_index
            if lidx in sideLanes:
                if lidx == currentLaneIndex:
                    classifiedSVs['current lane'].append(sv)
                else:
                    laneRelative = lidx[2] - currentLaneIndex[2]
                    if laneRelative == 1:
                        classifiedSVs['right lane'].append(sv)
                    elif laneRelative == -1:
                        classifiedSVs['left lane'].append(sv)
                    else:
                        continue
            elif lidx == nextLane:
                classifiedSVs['target lane'].append(sv)
            else:
                continue

        validVehicles: List[IDMVehicle] = []
        existVehicles: Dict[str, bool] = {}
        for k, v in classifiedSVs.items():
            if v:
                existVehicles[k] = True
            else:
                existVehicles[k] = False
            ahead, behind = self.processSingleLaneSVs(v)
            if ahead:
                validVehicles.append(ahead)
            if behind:
                validVehicles.append(behind)

        return validVehicles, existVehicles

    def describeSVNormalLane(self, currentLaneIndex: LaneIndex) -> str:
        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        surroundVehicles = self.getSurrendVehicles(10)
        validVehicles, existVehicles = self.processSVsNormalLane(
            surroundVehicles, currentLaneIndex
        )

        if not surroundVehicles:
            SVDescription = f"There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n"
            return SVDescription
        else:
            SVDescription = ''
            for sv in surroundVehicles:
                lidx = sv.lane_index
                if lidx in sideLanes:
                    if lidx == currentLaneIndex:
                        if sv in validVehicles:
                            SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on the same lane as you and {self.getSVRelativeState(sv)}. "
                        else:
                            continue
                    else:
                        laneRelative = lidx[2] - currentLaneIndex[2]
                        if laneRelative == 1:
                            if sv in validVehicles:
                                SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on the lane to your right and {self.getSVRelativeState(sv)}. "
                            else:
                                continue
                        elif laneRelative == -1:
                            if sv in validVehicles:
                                SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on the lane to your left and {self.getSVRelativeState(sv)}. "
                            else:
                                continue
                        else:
                            continue

                else:
                    continue
                SVDescription += f"The centre position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, length is {sv.LENGTH} meters, speed is {sv.speed:.2f} m/s, acceleration is {sv.action['acceleration']:.2f} m/s^2.\n"
            if SVDescription:
                descriptionPrefix = "There are other vehicles driving around you, and below is their basic information:\n"
                return descriptionPrefix + SVDescription
            else:
                SVDescription = f'There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n'
                return SVDescription

    def isInDangerousArea(self, sv: IDMVehicle) -> bool:
        relativeVector = sv.position - self.ego.position
        distance = np.linalg.norm(relativeVector)
        egoUnitVector = self.getUnitVector(self.ego.heading)
        relativeUnitVector = relativeVector / distance
        alpha = np.arccos(
            np.clip(np.dot(egoUnitVector, relativeUnitVector), -1, 1)
        )
        if alpha <= self.theta1:
            if distance <= self.radius1:
                return True
            else:
                return False
        elif self.theta1 < alpha <= self.theta2:
            if distance <= self.radius2:
                return True
            else:
                return False
        else:
            return False

    def evaluation(self, controlled_vehicle_count):
        env_avg_speed = 0
        controlled_vehicle_speed_list = []
        speed_list = []
        efficiency_score_list = []
        safety_score_list = []

        for vehicle in self.env.road.vehicles:
            speed_list.append(vehicle.speed)

        env_avg_speed = np.mean(speed_list)

        for i in range(controlled_vehicle_count):
            SingleLane_List = []
            TTC_ahead = 3
            TTC_behind = 3
            self.ego = self.env.controlled_vehicles[i]
            controlled_vehicle_speed = self.ego.speed
            controlled_vehicle_speed_list.append(controlled_vehicle_speed)
            efficiency_percent = controlled_vehicle_speed/env_avg_speed
            efficiency_score = efficiency_percent * 10
            if efficiency_score > 10:
                efficiency_score = 10
            efficiency_score = round(efficiency_score, 2)
            efficiency_score_list.append(efficiency_score)

            currentLaneIndex: LaneIndex = self.ego.lane_index
            surroundVehicles = self.getSurrendVehicles(10)
            for sv in surroundVehicles:
                if sv.lane_index == currentLaneIndex:
                    SingleLane_List.append(sv)
            ahead, behind = self.processSingleLaneSVs(SingleLane_List)
            if ahead :
                if ahead.speed < self.ego.speed:
                    TTC_ahead = (ahead.position[0] - self.ego.position[0]) / (self.ego.speed - ahead.speed)

            if behind :
                if behind.speed > self.ego.speed:
                    TTC_behind = (self.ego.position[0] - behind.position[0]) / (behind.speed - self.ego.speed)

            TTC = min(TTC_ahead, TTC_behind)
            if TTC >= 3:
                safety_score = 10
            elif TTC < 1.5:
                safety_score = 0
            else:
                safety_score = (TTC - 1.5) * 20 / 3
            safety_score = round(safety_score, 2)
            safety_score_list.append(safety_score)

        return controlled_vehicle_speed_list, efficiency_score_list, safety_score_list



    def describe(self, ego_number) -> str:
        self.ego = self.env.controlled_vehicles[ego_number]
        currentLaneIndex: LaneIndex = self.ego.lane_index
        print("lane_index",self.ego.lane_index)
        roadCondition = self.processNormalLane(currentLaneIndex)
        SVDescription = self.describeSVNormalLane(currentLaneIndex)

        return roadCondition + SVDescription


