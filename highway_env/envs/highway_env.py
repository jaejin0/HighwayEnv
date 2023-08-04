from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "heading", "cos_h", "sin_h"],
            },
            "action": {
                "type": "TrajectoryAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50, # 50
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            
            
            "reward_speed_range": [-40, 40],
            "normalize_reward": True,
            "offroad_terminal": False,
            
            
            ### Speed ###   25
            "speed_reward": 25,
            
            ### Safety ###   75
            "collision_reward": -50,
            "safe_distance_reward": 5,
            "on_road_reward": 20,
            
            "front_distance_range": [0, 30],
            "rear_distance_range": [0, 30],
            
            ### Energy Saving ###  0
            "torque_reward": 0
            
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        
        ### Speed ###
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        
        ### Safety ###
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        front_distance, rear_distance = self.vehicle.lane_distance_to(front_vehicle), self.vehicle.lane_distance_to(rear_vehicle)
        minimum_safe_distance = 30
        if front_distance > minimum_safe_distance:
            front_distance = minimum_safe_distance
        rear_distance = abs(rear_distance)

        front_distance = utils.lmap(front_distance, self.config["front_distance_range"], [0, 0.5])
        rear_distance = utils.lmap(rear_distance, self.config["rear_distance_range"], [0, 0.5])
        safe_distance = front_distance + rear_distance
        
        ### Energy Saving ###
        # finding acceleration
        ACCELERATION_RANGE = (-5, 5.0)
        acceleration = utils.lmap(self.action[0], [0, 1], ACCELERATION_RANGE)
        
        # config from class D sedan with electric powertrain from CarSim
        vehicle_mass = 1458  # [kg]
        rolling_resistance_coefficient = 0.012 
        gravity = 9.81  # [m/s^2]
        wheel_radius = 0.33  # [m]
        
        # config for assumed wheel mass
        wheel_mass = 20  # [kg]
        
        # calculation
        normal_force = vehicle_mass * gravity
        rolling_resistance = rolling_resistance_coefficient * normal_force  # rolling resistances for 4 wheels
        load_torque = rolling_resistance * wheel_radius
        
        angular_momentum = wheel_mass * self.vehicle.speed * wheel_radius
        angular_velocity = self.vehicle.speed / wheel_radius
        moment_of_inertia = angular_momentum / angular_velocity
        angular_acceleration = acceleration / wheel_radius
        print(acceleration)
        acceleration_torque = moment_of_inertia * angular_acceleration
        
        total_torque = acceleration_torque + load_torque
        
        # find total torque range and normalize
        # check if the equation is correct and find proofs for citations
        
        return {
            ### Speed ###
            "speed_reward": np.clip(scaled_speed, 0, 1),
            
            ### Safety ###
            "collision_reward": float(self.vehicle.crashed),
            "safe_distance_reward": np.clip(safe_distance, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            
            ### Energy Saving ###
            "torque_reward": np.clip(total_torque, 0, 1)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
