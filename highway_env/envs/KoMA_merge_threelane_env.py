from typing import Dict, Text

import numpy as np
import random
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class KoMAMergeThreeLaneEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "reward_speed_range": [20, 30],
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
            "other_vehicles_count": 5,
            "controlled_vehicles_count": 2,
            "random_seed": random.randint(1,9999)
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items())
        return utils.lmap(reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])

    def _rewards(self, action: int) -> Dict[Text, float]:
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": scaled_speed,
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle)
            )
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        #print("crash" + str(self.vehicle.crashed))
        #print("over"  + str(self.vehicle.position[0] > 370))
        is_terminated = False
        for i in range(self.config["controlled_vehicles_count"]):
            is_terminate = (self.controlled_vehicles[i].crashed | bool(self.controlled_vehicles[i].position[0] > 450))
            is_terminated = (is_terminate | is_terminated)
        return is_terminated

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 120, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [-StraightLane.DEFAULT_WIDTH, 0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, s], [n, c]]
        line_type_merge = [[c, s], [n, s], [n, s]]
        for i in range(3):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        self.controlled_vehicles = []
        road = self.road
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        spawn_points_s = [[160, 0], [160, 1], [160, 2], [200, 0], [200, 1], [200, 2], [240, 0], [240, 1], [240, 2], [280, 0], [280, 1], [280, 2]] #150 230 310 460
        spawn_points_m = [5, 45, 85, 125, 165, 205]

        # initial location noise
        loc_noise = np.random.rand(self.config["other_vehicles_count"] + self.config["controlled_vehicles_count"] -1) * 20 - 10  # range from [-10, 10]
        loc_noise = list(loc_noise)
        speed_noise = np.random.rand(self.config["other_vehicles_count"] + self.config["controlled_vehicles_count"] ) * 5  # range from [0, 5]
        speed_noise = list(speed_noise)

        spawn_point_s_c = random.sample(spawn_points_s, self.config["controlled_vehicles_count"] - 1)
        for a in spawn_point_s_c:
            spawn_points_s.remove(a)

        for _ in spawn_point_s_c:
            init_location = spawn_point_s_c.pop(0)
            if init_location[1] != 3:
                ego_vehicle = self.action_type.vehicle_class(road,
                                                      road.network.get_lane(("a", "b", init_location[1])).position(
                                                          init_location[0] + loc_noise.pop(0), 0),
                                                      speed=20 + speed_noise.pop(0))
            else:
                ego_vehicle = self.action_type.vehicle_class(road,
                                                             road.network.get_lane(
                                                                 ("k", "b", 0)).position(
                                                                 init_location[0] + loc_noise.pop(0) - 150, 0),
                                                             speed=20 + speed_noise.pop(0))
            road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)

        # spawn point indexes on the straight road
        spawn_point_s_h = random.sample(spawn_points_s, self.config["other_vehicles_count"])


        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("b", "c", 3)).position(1, 0),
                                                     speed=17.5 + speed_noise.pop(0))
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)


        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["other_vehicles_count"]):
            init_location = spawn_point_s_h.pop(0)
            if init_location[1] != 3:
                other_vehicle = other_vehicles_type(road, road.network.get_lane(("a", "b", init_location[1])).position(
                    init_location[0] + loc_noise.pop(0), 0), speed=20 + speed_noise.pop(0))
            else:
                other_vehicle = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(
                    init_location[0] + loc_noise.pop(0) - 150, 0), speed=20 + speed_noise.pop(0))
            road.vehicles.append(other_vehicle)

        
