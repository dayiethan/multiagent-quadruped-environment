from collections import OrderedDict, defaultdict
import itertools
import numpy as np
from isaacgym.torch_utils import torch_rand_float, get_euler_xyz, quat_from_euler_xyz, tf_apply
from isaacgym import gymtorch, gymapi, gymutil
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import get_terrain_cls
from ..base.legged_robot_config import LeggedRobotCfg

class LeggedRobotField(LeggedRobot):
    """ NOTE: Most of this class implementation does not depend on the terrain. Check where
    `check_BarrierTrack_terrain` is called to remove the dependency of BarrierTrack terrain.
    """
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        print("Using LeggedRobotField.__init__, num_obs and num_privileged_obs will be computed instead of assigned.")
        # cfg.terrain.measure_heights = True # force height measurement that have full obs from parent class implementation.
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    
    ##### adds-on with sensors #####
    def _create_sensors(self, env_handle=None, actor_handle= None):
        sensor_handle_dict = super()._create_sensors(
            env_handle= env_handle,
            actor_handle= actor_handle,
        )
        all_obs_components = self.all_obs_components

        if "forward_depth" in all_obs_components or "forward_color" in all_obs_components:
            camera_handle = self._create_onboard_camera(env_handle, actor_handle, "forward_camera")
            sensor_handle_dict["forward_camera"] = camera_handle

        return sensor_handle_dict

    def _create_onboard_camera(self, env_handle, actor_handle, sensor_name):
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = getattr(self.cfg.sensor, sensor_name).resolution[0]
        camera_props.width = getattr(self.cfg.sensor, sensor_name).resolution[1]
        if hasattr(getattr(self.cfg.sensor, sensor_name), "horizontal_fov"):
            camera_props.horizontal_fov = np.random.uniform(
                getattr(self.cfg.sensor, sensor_name).horizontal_fov[0],
                getattr(self.cfg.sensor, sensor_name).horizontal_fov[1],
            ) if isinstance(getattr(self.cfg.sensor, sensor_name).horizontal_fov, (tuple, list)) else getattr(self.cfg.sensor, sensor_name).horizontal_fov
            # vertical_fov = horizontal_fov * camera_props.height / camera_props.width
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        local_transform = gymapi.Transform()
        if isinstance(getattr(self.cfg.sensor, sensor_name).position, dict):
            # allow domain randomization across robots.
            # sample from "mean" and "std" attributes.
            # each must be a list of 3 elements.
            cam_x = np.random.normal(
                getattr(self.cfg.sensor, sensor_name).position["mean"][0],
                getattr(self.cfg.sensor, sensor_name).position["std"][0],
            )
            cam_y = np.random.normal(
                getattr(self.cfg.sensor, sensor_name).position["mean"][1],
                getattr(self.cfg.sensor, sensor_name).position["std"][1],
            )
            cam_z = np.random.normal(
                getattr(self.cfg.sensor, sensor_name).position["mean"][2],
                getattr(self.cfg.sensor, sensor_name).position["std"][2],
            )
            local_transform.p = gymapi.Vec3(cam_x, cam_y, cam_z)
        else:
            local_transform.p = gymapi.Vec3(*getattr(self.cfg.sensor, sensor_name).position)
        if isinstance(getattr(self.cfg.sensor, sensor_name).rotation, dict):
            # allow domain randomization across robots
            # sample from "lower" and "upper" attributes.
            # each must be a list of 3 elements (in radian).
            cam_roll = np.random.uniform(0, 1) * (
                getattr(self.cfg.sensor, sensor_name).rotation["upper"][0] - \
                getattr(self.cfg.sensor, sensor_name).rotation["lower"][0]
            ) + getattr(self.cfg.sensor, sensor_name).rotation["lower"][0]
            cam_pitch = np.random.uniform(0, 1) * (
                getattr(self.cfg.sensor, sensor_name).rotation["upper"][1] - \
                getattr(self.cfg.sensor, sensor_name).rotation["lower"][1]
            ) + getattr(self.cfg.sensor, sensor_name).rotation["lower"][1]
            cam_yaw = np.random.uniform(0, 1) * (
                getattr(self.cfg.sensor, sensor_name).rotation["upper"][2] - \
                getattr(self.cfg.sensor, sensor_name).rotation["lower"][2]
            ) + getattr(self.cfg.sensor, sensor_name).rotation["lower"][2]
            local_transform.r = gymapi.Quat.from_euler_zyx(cam_yaw, cam_pitch, cam_roll)
        else:
            local_transform.r = gymapi.Quat.from_euler_zyx(*getattr(self.cfg.sensor, sensor_name).rotation)
        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        
        return camera_handle

    ##### Working on simulation steps #####
    def pre_physics_step(self, actions):
        self.volume_sample_points_refreshed = False
        actions_preprocessed = False
        if isinstance(self.cfg.normalization.clip_actions, (tuple, list)):
            self.cfg.normalization.clip_actions = torch.tensor(
                self.cfg.normalization.clip_actions,
                device= self.device,
            )
        if getattr(self.cfg.normalization, "clip_actions_method", None) == "tanh":
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = (torch.tanh(actions) * clip_actions).to(self.device)
            actions_preprocessed = True
        if getattr(self.cfg.normalization, "clip_actions_delta", None) is not None:
            self.actions = torch.clip(
                self.actions,
                self.last_actions - self.cfg.normalization.clip_actions_delta,
                self.last_actions + self.cfg.normalization.clip_actions_delta,
            )
        
        if not actions_preprocessed:
            return super().pre_physics_step(actions)
    
    def post_physics_step(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        return super().post_physics_step()
    
    def check_termination(self):
        return_ = super().check_termination()
        if not hasattr(self.cfg, "termination"): return return_
        
        r, p, y = get_euler_xyz(self.base_quat)
        r[r > np.pi] -= np.pi * 2 # to range (-pi, pi)
        p[p > np.pi] -= np.pi * 2 # to range (-pi, pi)
        z = self.root_states[:, 2] - self.agent_origins.reshape(-1, 3)[:, 2]

        # if getattr(self.cfg.termination, "check_obstacle_conditioned_threshold", False) and self.check_BarrierTrack_terrain():
        #     if hasattr(self, "volume_sample_points"):
        #         self.refresh_volume_sample_points()
        #         stepping_obstacle_info = self.terrain.get_stepping_obstacle_info(self.volume_sample_points.view(-1, 3))
        #     else:
        #         stepping_obstacle_info = self.terrain.get_stepping_obstacle_info(self.root_states[:, :3])
        #     stepping_obstacle_info = stepping_obstacle_info.view(self.num_envs, -1, stepping_obstacle_info.shape[-1])
        #     # Assuming that each robot will only be in one obstacle or non obstacle.
        #     robot_stepping_obstacle_id = torch.max(stepping_obstacle_info[:, :, 0], dim= -1)[0]
        
        if "roll" in self.cfg.termination.termination_terms:
            if "robot_stepping_obstacle_id" in locals():
                raise NotImplementedError
                r_term_buff = torch.abs(r[robot_stepping_obstacle_id == 0]) > \
                    self.cfg.termination.roll_kwargs["threshold"]
                self.reset_buf[robot_stepping_obstacle_id == 0] |= r_term_buff
                for obstacle_name, obstacle_id in self.terrain.track_options_id_dict.items():
                    if (obstacle_name + "_threshold") in self.cfg.termination.roll_kwargs:
                        env_selection_mask = robot_stepping_obstacle_id == obstacle_id
                        r_term_buff = torch.abs(r[env_selection_mask]) > \
                            self.cfg.termination.roll_kwargs[obstacle_name + "_threshold"]
                        self.reset_buf[env_selection_mask] |= r_term_buff
            else:
                r_term_buff = (torch.abs(r) > self.cfg.termination.roll_kwargs["threshold"]).reshape(self.num_envs, -1).sum(1).to(torch.bool)
                self.reset_buf |= r_term_buff
        if "pitch" in self.cfg.termination.termination_terms:
            if "robot_stepping_obstacle_id" in locals():
                raise NotImplementedError
                p_term_buff = torch.abs(p[robot_stepping_obstacle_id == 0]) > \
                    self.cfg.termination.pitch_kwargs["threshold"]
                self.reset_buf[robot_stepping_obstacle_id == 0] |= p_term_buff
                for obstacle_name, obstacle_id in self.terrain.track_options_id_dict.items():
                    if (obstacle_name + "_threshold") in self.cfg.termination.pitch_kwargs:
                        env_selection_mask = robot_stepping_obstacle_id == obstacle_id
                        p_term_buff = torch.abs(p[env_selection_mask]) > \
                            self.cfg.termination.pitch_kwargs[obstacle_name + "_threshold"]
                        self.reset_buf[env_selection_mask] |= p_term_buff
            else:
                p_term_buff = (torch.abs(p) > self.cfg.termination.pitch_kwargs["threshold"]).reshape(self.num_envs, -1).sum(1).to(torch.bool)
                self.reset_buf |= p_term_buff
        if "z_low" in self.cfg.termination.termination_terms:
            if "robot_stepping_obstacle_id" in locals():
                raise NotImplementedError
                z_low_term_buff = z[robot_stepping_obstacle_id == 0] < \
                    self.cfg.termination.z_low_kwargs["threshold"]
                self.reset_buf[robot_stepping_obstacle_id == 0] |= z_low_term_buff
                for obstacle_name, obstacle_id in self.terrain.track_options_id_dict.items():
                    if (obstacle_name + "_threshold") in self.cfg.termination.z_low_kwargs:
                        env_selection_mask = robot_stepping_obstacle_id == obstacle_id
                        z_low_term_buff = z[env_selection_mask] < \
                            self.cfg.termination.z_low_kwargs[obstacle_name + "_threshold"]
                        self.reset_buf[env_selection_mask] |= z_low_term_buff
            else:
                z_low_term_buff = (z < self.cfg.termination.z_low_kwargs["threshold"]).reshape(self.num_envs, -1).sum(1).to(torch.bool)
                self.reset_buf |= z_low_term_buff
        if "z_high" in self.cfg.termination.termination_terms:
            z_high_term_buff = (z > self.cfg.termination.z_high_kwargs["threshold"]).reshape(self.num_envs, -1).sum(1).to(torch.bool)
            self.reset_buf |= z_high_term_buff
        # if "out_of_track" in self.cfg.termination.termination_terms and self.check_BarrierTrack_terrain():
        #     # robot considered dead if it goes side ways
        #     side_distance = self.terrain.get_sidewall_distance(self.root_states[:, :3])
        #     side_diff = torch.abs(side_distance[..., 0] - side_distance[..., 1])
        #     out_of_track_buff = side_diff > self.cfg.termination.out_of_track_kwargs["threshold"]
        #     out_of_track_buff |= side_distance[..., 0] <= 0
        #     out_of_track_buff |= side_distance[..., 1] <= 0
        #     self.reset_buf |= out_of_track_buff.reshape(self.num_envs, -1).sum(1).to(torch.bool)

        # if getattr(self.cfg.termination, "timeout_at_border", False) and self.check_BarrierTrack_terrain():
        #     track_idx = self.terrain.get_track_idx(self.root_states[:, :3], clipped= False)
        #     # The robot is going +x direction, so no checking for row_idx <= 0
        #     out_of_border_buff = track_idx[:, 0] >= self.terrain.cfg.num_rows
        #     out_of_border_buff |= track_idx[:, 1] < 0
        #     out_of_border_buff |= track_idx[:, 1] >= self.terrain.cfg.num_cols
        #     out_of_border_buff = out_of_border_buff.reshape(self.num_envs, -1).sum(1).to(torch.bool)

        #     self.time_out_buf |= out_of_border_buff
        #     self.reset_buf |= out_of_border_buff
        # if getattr(self.cfg.termination, "timeout_at_finished", False) and self.check_BarrierTrack_terrain():
        #     raise NotImplementedError
        #     # TODO: create finish line
        #     x = self.root_states[:, 0] - self.env_origins[:, 0]
        #     finished_buffer = x > (self.terrain.env_length)
        #     self.time_out_buf |= finished_buffer
        #     self.reset_buf |= finished_buffer
        
        return return_

    def _fill_extras(self, env_ids):
        return_ = super()._fill_extras(env_ids)

        self.extras["episode"]["max_pos_x"] = 0.
        self.extras["episode"]["min_pos_x"] = 0.
        self.extras["episode"]["max_pos_y"] = 0.
        self.extras["episode"]["min_pos_y"] = 0.
        # self.extras["episode"]["n_obstacle_passed"] = 0.
        with torch.no_grad():
            pos_x = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
            self.extras["episode"]["pos_x"] = pos_x
            # if self.check_BarrierTrack_terrain():
            #     self.extras["episode"]["n_obstacle_passed"] = None
        
        return return_

    def _post_physics_step_callback(self):
        return_ = super()._post_physics_step_callback()

        with torch.no_grad():
            pos_x = self.root_states[:, 0] - self.agent_origins.reshape(-1, 3)[:, 0]
            pos_y = self.root_states[:, 1] - self.agent_origins.reshape(-1, 3)[:, 1]
            self.extras["episode"]["max_pos_x"] = max(self.extras["episode"]["max_pos_x"], torch.max(pos_x).cpu())
            self.extras["episode"]["min_pos_x"] = min(self.extras["episode"]["min_pos_x"], torch.min(pos_x).cpu())
            self.extras["episode"]["max_pos_y"] = max(self.extras["episode"]["max_pos_y"], torch.max(pos_y).cpu())
            self.extras["episode"]["min_pos_y"] = min(self.extras["episode"]["min_pos_y"], torch.min(pos_y).cpu())
            # if self.check_BarrierTrack_terrain():
            #     self.extras["episode"]["n_obstacle_passed"] = None

        return return_
    
    def _compute_torques(self, actions):
        if hasattr(self, "motor_strength"):
            actions = self.motor_strength * actions
        return super()._compute_torques(actions)
    
    def _get_terrain_curriculum_move(self, env_ids):
        if not (self.cfg.terrain.selected == "BarrierTrack" and self.cfg.terrain.BarrierTrack_kwargs["virtual_terrain"] and hasattr(self, "body_sample_indices")):
            if getattr(self.cfg.curriculum, "no_moveup_when_fall", False):
                move_up, move_down = super()._get_terrain_curriculum_move(env_ids)
                move_up = move_up & self.time_out_buf[env_ids]
                return move_up, move_down
            else:
                return super()._get_terrain_curriculum_move(env_ids)
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        moved = distance > (self.terrain.env_block_length * 1.5) # 0.1 is the guess of robot touching the obstacle block.
        passed_depths = self.terrain.get_passed_obstacle_depths(
            self.terrain_levels[env_ids],
            self.terrain_types[env_ids],
            self.volume_sample_points[env_ids, :, 0].max(-1)[0], # choose the sample points that goes the furthest
        ) + 1e-12

        p_v_ok = p_d_ok = 1
        p_v_too_much = p_d_too_much = 0
        # NOTE: only when penetrate_* reward is computed does this function check the penetration
        if "penetrate_volume" in self.episode_sums:
            p_v = self.episode_sums["penetrate_volume"][env_ids]
            p_v_normalized = p_v / passed_depths / self.reward_scales["penetrate_volume"]
            p_v_ok = p_v_normalized < self.cfg.curriculum.penetrate_volume_threshold_harder
            p_v_too_much = p_v_normalized > self.cfg.curriculum.penetrate_volume_threshold_easier
        if "penetrate_depth" in self.episode_sums:
            p_d = self.episode_sums["penetrate_depth"][env_ids]
            p_d_normalized = p_d / passed_depths / self.reward_scales["penetrate_depth"]
            p_d_ok = p_d_normalized < self.cfg.curriculum.penetrate_depth_threshold_harder
            p_d_too_much = p_d_normalized > self.cfg.curriculum.penetrate_depth_threshold_easier

        # print("p_v:", p_v_normalized, "p_d:", p_d_normalized)
        move_up = p_v_ok * p_d_ok * moved
        move_down = ((~moved) + p_v_too_much + p_d_too_much).to(bool)
        return move_up, move_down

    ##### Dealing with observations #####
    def _init_buffers(self):
        # update obs_scales components incase there will be one-by-one scaling
        for k in self.all_obs_components:
            if isinstance(getattr(self.obs_scales, k, None), (tuple, list)):
                setattr(
                    self.obs_scales,
                    k,
                    torch.tensor(getattr(self.obs_scales, k, 1.), dtype= torch.float32, device= self.device)
                )
        
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.all_rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        # add sensor dict, which will be filled during create sensor
        self.sensor_tensor_dict = defaultdict(list)

        for env_i, env_handle in enumerate(self.envs):
            if "forward_depth" in self.all_obs_components:
                env_sensor_tensors = []
                for agent_i in range(self.num_agents):
                    env_sensor_tensors.append(gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(
                            self.sim,
                            env_handle,
                            self.sensor_handles[env_i][agent_i]["forward_camera"],
                            gymapi.IMAGE_DEPTH,
                    )))
                self.sensor_tensor_dict["forward_depth"].append(torch.stack(env_sensor_tensors))
            if "forward_color" in self.all_obs_components:
                env_sensor_tensors = []
                for agent_i in range(self.num_agents):
                    env_sensor_tensors.append(gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(
                            self.sim,
                            env_handle,
                            self.sensor_handles[env_i][agent_i]["forward_camera"],
                            gymapi.IMAGE_COLOR,
                    )))
                self.sensor_tensor_dict["forward_color"].append(torch.stack(env_sensor_tensors))

    def _reset_buffers(self, env_ids):
        return_ = super()._reset_buffers(env_ids)
        if hasattr(self, "velocity_sample_points"): self.velocity_sample_points[env_ids] = 0.
        return return_
        
    def _prepare_reward_function(self):
        return_ = super()._prepare_reward_function()

        # get the body indices within the simulation (for estimating robot state)
        if "penetrate_volume" in self.reward_names or "penetrate_depth" in self.reward_names:
            self._init_volume_sample_points()
            print("Total number of volume estimation points for each robot is:", self.volume_sample_points.shape[1])

        return return_

    def _init_volume_sample_points(self):
        """ Build sample points for penetration volume estimation
        NOTE: self.cfg.sim.body_measure_points must be a dict with
            key: body name (or part of the body name) to estimate
            value: dict(
                x, y, z: sample points to form a meshgrid
                transform: [x, y, z, roll, pitch, yaw] for transforming the meshgrid w.r.t body frame
            )
        """
        # read and specify the order of which body to sample from and its relative sample points.
        self.body_measure_name_order = [] # order specified
        self.body_sample_indices = []
        for idx in range(self.num_envs):
            rigid_body_names = self.gym.get_actor_rigid_body_names(self.envs[idx], self.actor_handles[idx])
            self.body_sample_indices.append([])
            for name, measure_name in itertools.product(rigid_body_names, self.cfg.sim.body_measure_points.keys()):
                if measure_name in name:
                    self.body_sample_indices[-1].append(
                        self.gym.find_actor_rigid_body_index(
                            self.envs[idx],
                            self.actor_handles[idx],
                            name,
                            gymapi.IndexDomain.DOMAIN_SIM,
                    ))
                    if idx == 0: # assuming all envs have the same actor configuration
                        self.body_measure_name_order.append(measure_name) # order specified
        self.body_sample_indices = torch.tensor(self.body_sample_indices, device= self.sim_device).flatten() # n_envs * num_bodies

        # compute and store each sample points in body frame.
        self.body_volume_points = dict()
        for measure_name, points_cfg in self.cfg.sim.body_measure_points.items():
            x = torch.tensor(points_cfg["x"], device= self.device, dtype= torch.float32, requires_grad= False)
            y = torch.tensor(points_cfg["y"], device= self.device, dtype= torch.float32, requires_grad= False)
            z = torch.tensor(points_cfg["z"], device= self.device, dtype= torch.float32, requires_grad= False)
            t = torch.tensor(points_cfg["transform"][0:3], device= self.device, dtype= torch.float32, requires_grad= False)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
            grid_points = torch.stack([
                grid_x.flatten(),
                grid_y.flatten(),
                grid_z.flatten(),
            ], dim= -1) # n_points, 3
            q = quat_from_euler_xyz(
                torch.tensor(points_cfg["transform"][3], device= self.sim_device, dtype= torch.float32),
                torch.tensor(points_cfg["transform"][4], device= self.sim_device, dtype= torch.float32),
                torch.tensor(points_cfg["transform"][5], device= self.sim_device, dtype= torch.float32),
            )
            self.body_volume_points[measure_name] = tf_apply(
                q.expand(grid_points.shape[0], 4),
                t.expand(grid_points.shape[0], 3),
                grid_points,
            )
        num_sample_points_per_env = 0
        for body_name in self.body_measure_name_order:
            for measure_name in self.body_volume_points.keys():
                if measure_name in body_name:
                    num_sample_points_per_env += self.body_volume_points[measure_name].shape[0]
        self.volume_sample_points = torch.zeros(
            (self.num_envs, num_sample_points_per_env, 3),
            device= self.device,
            dtype= torch.float32,    
        )
        self.velocity_sample_points = torch.zeros(
            (self.num_envs, num_sample_points_per_env, 3),
            device= self.device,
            dtype= torch.float32,    
        )

    def _get_proprioception_obs(self, privileged= False):
        return self.obs_super_impl[:, :48]
    
    def _get_height_measurements_obs(self, privileged= False):
        return self.obs_super_impl[:, 48:235]

    def _get_forward_depth_obs(self, privileged= False):
        return torch.cat(self.sensor_tensor_dict["forward_depth"], dim=0).flatten(start_dim= 1)

    def _get_base_pose_obs(self, privileged= False):
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        roll[roll > np.pi] -= np.pi * 2 # to range (-pi, pi)
        pitch[pitch > np.pi] -= np.pi * 2 # to range (-pi, pi)
        yaw[yaw > np.pi] -= np.pi * 2 # to range (-pi, pi)
        return torch.cat([
            self.root_states[:, :3] - self.agent_origins.reshape(-1, 3),
            torch.stack([roll, pitch, yaw], dim= -1),
        ], dim= -1)
    
    def _get_robot_config_obs(self, privileged= False):
        return self.robot_config_buffer

    def _get_engaging_block_obs(self, privileged= False):
        """ Compute the obstacle info for the robot """
        if not self.check_BarrierTrack_terrain():
            # This could be wrong, check BarrierTrack implementation to get the exact shape.
            return torch.zeros((self.num_envs, (1 + (4 + 1) + 2)), device= self.sim_device)
        base_positions = self.root_states[:, 0:3] # (n_envs, 3)
        self.refresh_volume_sample_points()
        return self.terrain.get_engaging_block_info(
            base_positions,
            self.volume_sample_points - base_positions.unsqueeze(-2), # (n_envs, n_points, 3)
        )

    def _get_sidewall_distance_obs(self, privileged= False):
        if not self.check_BarrierTrack_terrain():
            return torch.zeros((self.num_envs, 2), device= self.sim_device)
        base_positions = self.root_states[:, 0:3] # (n_envs, 3)
        return self.terrain.get_sidewall_distance(base_positions)

    def _get_obs_from_components(self, components: list, privileged= False):
        obs_segments = self.get_obs_segment_from_components(components)
        obs = []
        for k, v in obs_segments.items():
            if k == "proprioception":
                obs.append(self._get_proprioception_obs(privileged))
            elif k == "height_measurements":
                obs.append(self._get_height_measurements_obs(privileged))
            else:
                # get the observation from specific component name
                # such as "_get_forward_depth_obs"
                obs.append(
                    getattr(self, "_get_" + k + "_obs")(privileged) * \
                    getattr(self.obs_scales, k, 1.)
                )
        obs = torch.cat(obs, dim=1)
        return obs

    def compute_observations(self):
        for key in self.sensor_handles[0][0].keys():
            if "camera" in key:
                # NOTE: Different from the documentation and examples from isaacgym
                # gym.fetch_results() must be called before gym.start_access_image_tensors()
                # refer to https://forums.developer.nvidia.com/t/camera-example-and-headless-mode/178901/10
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                break
        add_noise = self.add_noise; self.add_noise = False
        return_ = super().compute_observations() # currently self.obs_buf is a mess
        self.obs_super_impl = self.obs_buf
        self.add_noise = add_noise

        # actor obs
        self.obs_buf = self._get_obs_from_components(
            self.cfg.env.obs_components,
            privileged= False,
        )
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # critic obs
        if not self.num_privileged_obs is None:
            self.privileged_obs_buf[:] = self._get_obs_from_components(
                self.cfg.env.privileged_obs_components,
                privileged= getattr(self.cfg.env, "privileged_obs_gets_privilege", False),
            )
        # fixing linear velocity in proprioception observation
        if "proprioception" in getattr(self.cfg.env, "privileged_obs_components", []) \
            and getattr(self.cfg.env, "privileged_use_lin_vel", False):
            # NOTE: according to self.get_obs_segment_from_components, "proprioception" observation
            # is always the first part of this flattened observation. check super().compute_observations
            # and self.cfg.env.use_lin_vel for the reason of this if branch.
            self.privileged_obs_buf[:, :3] = self.base_lin_vel * self.obs_scales.lin_vel

        for key in self.sensor_handles[0][0].keys():
            if "camera" in key:
                self.gym.end_access_image_tensors(self.sim)
                break
        return return_

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        
        segment_start_idx = 0
        obs_segments = self.get_obs_segment_from_components(cfg.env.obs_components)
        # write noise for each corresponding component.
        for k, v in obs_segments.items():
            segment_length = np.prod(v)
            # write sensor scale to provided noise_vec
            # for example "_write_forward_depth_noise"
            getattr(self, "_write_" + k + "_noise")(noise_vec[segment_start_idx: segment_start_idx + segment_length])
            segment_start_idx += segment_length

        return noise_vec

    def _write_forward_depth_noise(self, noise_vec):
        noise_vec[:] = self.cfg.noise.noise_scales.forward_depth * self.cfg.noise.noise_level * self.obs_scales.forward_depth

    def _write_base_pose_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "base_pose"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.base_pose * self.cfg.noise.noise_level * self.obs_scales.base_pose
    
    def _write_robot_config_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "robot_config"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.robot_config * self.cfg.noise.noise_level * self.obs_scales.robot_config

    def _write_engaging_block_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "engaging_block"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.engaging_block * self.cfg.noise.noise_level * self.obs_scales.engaging_block
    
    def _write_sidewall_distance_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "sidewall_distance"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.sidewall_distance * self.cfg.noise.noise_level * self.obs_scales.sidewall_distance

    ##### adds-on with building the environment #####
    def _create_terrain(self):
        """ Using cfg.terrain.selected to identify terrain class """
        if not isinstance(self.cfg.terrain.selected, str):
            return super()._create_terrain()
        terrain_cls = self.cfg.terrain.selected
        self.terrain = get_terrain_cls(terrain_cls)(self.cfg.terrain, self.num_envs, self.num_agents)
        self.terrain.add_terrain_to_sim(self.gym, self.sim, self.device)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength = torch_rand_float(
                self.cfg.domain_rand.leg_motor_strength_range[0],
                self.cfg.domain_rand.leg_motor_strength_range[1],
                (self.num_envs, 12 * self.num_agents),
                device=self.device,
            )
        return super()._create_envs()
    
    def _process_rigid_shape_props(self, props, env_id):
        props = super()._process_rigid_shape_props(props, env_id)
        if env_id == 0:
            all_obs_components = self.all_obs_components
            if "robot_config" in all_obs_components:
                # all_obs_components
                self.robot_config_buffer = torch.empty(
                    self.num_envs * self.num_agents, 1 + 3 + 1 + 12,
                    dtype= torch.float32,
                    device= self.device,
                )
        
        if hasattr(self, "robot_config_buffer"):
            self.robot_config_buffer[env_id, 0] = props[0].friction
        return props

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        if env_id == 0:
            if hasattr(self.cfg.control, "torque_limits"):
                if not isinstance(self.cfg.control.torque_limits, (tuple, list)):
                    self.torque_limits = torch.ones(self.num_actuated_dof, dtype= torch.float, device= self.device, requires_grad= False)
                    self.torque_limits *= self.cfg.control.torque_limits
                else:
                    assert not self.num_actuated_dof % len(self.cfg.control.torque_limits), "torque_limits does not fit num_dof"
                    self.torque_limits = torch.tensor(self.cfg.control.torque_limits * (self.num_actuated_dof // len(self.cfg.control.torque_limits)), dtype= torch.float, device= self.device, requires_grad= False)
        return props

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)

        if self.cfg.domain_rand.randomize_com:
            rng_com_x = self.cfg.domain_rand.com_range.x
            rng_com_y = self.cfg.domain_rand.com_range.y
            rng_com_z = self.cfg.domain_rand.com_range.z
            rand_com = np.random.uniform(
                [rng_com_x[0], rng_com_y[0], rng_com_z[0]],
                [rng_com_x[1], rng_com_y[1], rng_com_z[1]],
                size=(3,),
            )
            props[0].com += gymapi.Vec3(*rand_com)

        agent_id = self.env_agent_indices[env_id].reshape(-1)

        if hasattr(self, "robot_config_buffer"):
            self.robot_config_buffer[agent_id, 1] = props[0].com.x
            self.robot_config_buffer[agent_id, 2] = props[0].com.y
            self.robot_config_buffer[agent_id, 3] = props[0].com.z
            self.robot_config_buffer[agent_id, 4] = props[0].mass
            self.robot_config_buffer[agent_id, 5 : 5 + 12] = self.motor_strength[env_id].reshape(self.num_agents, -1) if hasattr(self, "motor_strength") else 1.
        return props

    def _get_env_origins(self):
        super()._get_env_origins()
        self.custom_origins = True

    def _draw_sensor_vis(self, env_h, sensor_hd):
        for sensor_name, sensor_h in sensor_hd.items():
            if "camera" in sensor_name:
                camera_transform = self.gym.get_camera_transform(self.sim, env_h, sensor_h)
                cam_axes = gymutil.AxesGeometry(scale= 0.1)
                gymutil.draw_lines(cam_axes, self.gym, self.viewer, env_h, camera_transform)

    # def _draw_debug_vis(self):
    #     if not "height_measurements" in self.all_obs_components:
    #         measure_heights_tmp = self.terrain.cfg.measure_heights
    #         self.terrain.cfg.measure_heights = False
    #         return_ = super()._draw_debug_vis()
    #         self.terrain.cfg.measure_heights = measure_heights_tmp
    #     else:
    #         return_ = super()._draw_debug_vis()
    #     if self.cfg.terrain.selected == "BarrierTrack":
    #         self.terrain.draw_virtual_terrain(self.viewer)
    #     if hasattr(self, "volume_sample_points") and self.cfg.viewer.draw_volume_sample_points:
    #         self.draw_volume_sample_points()
    #     for env_h, sensor_hd in zip(self.envs, self.sensor_handles):
    #         self._draw_sensor_vis(env_h, sensor_hd)
    #     return return_

    ##### defines observation segments, which tells the order of the entire flattened obs #####
    def get_obs_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict()
        if "proprioception" in components:
            segments["proprioception"] = (48,)
        if "height_measurements" in components:
            segments["height_measurements"] = (187,)
        if "forward_depth" in components:
            segments["forward_depth"] = (1, *self.cfg.sensor.forward_camera.resolution)
        if "base_pose" in components:
            segments["base_pose"] = (6,) # xyz + rpy
        if "robot_config" in components:
            """ Related to robot_config_buffer attribute, Be careful to change. """
            # robot shape friction
            # CoM (Center of Mass) x, y, z
            # base mass (payload)
            # motor strength for each joint
            segments["robot_config"] = (1 + 3 + 1 + 12,)
        if "engaging_block" in components:
            if not self.check_BarrierTrack_terrain():
                # This could be wrong, please check the implementation of BarrierTrack
                segments["engaging_block"] = (1 + (4 + 1) + 2,)
            else:
                segments["engaging_block"] = get_terrain_cls("BarrierTrack").get_engaging_block_info_shape(self.cfg.terrain)
        if "sidewall_distance" in components:
            self.check_BarrierTrack_terrain()
            segments["sidewall_distance"] = (2,)
        return segments

    def get_num_obs_from_components(self, components):
        obs_segments = self.get_obs_segment_from_components(components)
        num_obs = 0
        for k, v in obs_segments.items():
            num_obs += np.prod(v)
        return num_obs

    def refresh_volume_sample_points(self):
        if self.volume_sample_points_refreshed:
            return
        sampled_body_pos = self.all_rigid_body_states[self.body_sample_indices, :3].view(self.num_envs, -1, 3)
        sampled_body_quat = self.all_rigid_body_states[self.body_sample_indices, 3:7].view(self.num_envs, -1, 4)
        sample_points_start_idx = 0
        for body_idx, body_measure_name in enumerate(self.body_measure_name_order):
            num_volume_points = self.body_volume_points[body_measure_name].shape[0]                
            point_positions = tf_apply(
                    sampled_body_quat[:, body_idx].unsqueeze(1).expand(-1, num_volume_points, -1),
                    sampled_body_pos[:, body_idx].unsqueeze(1).expand(-1, num_volume_points, -1),
                    self.body_volume_points[body_measure_name].unsqueeze(0).expand(self.num_envs, -1, -1),
                ) # (num_envs, num_volume_points, 3)
            valid_velocity_mask = self.episode_length_buf > 0
            self.velocity_sample_points[valid_velocity_mask, sample_points_start_idx: sample_points_start_idx + num_volume_points] = \
                (point_positions[valid_velocity_mask] - self.volume_sample_points[valid_velocity_mask, sample_points_start_idx: sample_points_start_idx + num_volume_points]) / self.dt
            self.volume_sample_points[:, sample_points_start_idx: sample_points_start_idx + num_volume_points] = point_positions
            sample_points_start_idx += num_volume_points
        self.volume_sample_points_refreshed = True

    def draw_volume_sample_points(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 4, 4, None, color=(1, 0.1, 0))
        for env_idx in range(self.num_envs):
            for point_idx in range(self.volume_sample_points.shape[1]):
                sphere_pose = gymapi.Transform(gymapi.Vec3(*self.volume_sample_points[env_idx, point_idx]), r= None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_idx], sphere_pose)

    ##### Additional rewards #####
    def _reward_lin_vel_l2norm(self):
        return torch.norm((self.commands[:, :2] - self.base_lin_vel[:, :2]), dim= 1)

    def _reward_world_vel_l2norm(self):
        return torch.norm((self.commands[:, :2] - self.root_states[:, 7:9]), dim= 1)

    def _reward_legs_energy(self):
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)

    def _reward_legs_energy_substeps(self):
        # (n_envs, n_substeps, n_dof)
        # reshape -> (n_agents, n_substeps, n_dofs)
        # square sum -> (n_envs, n_substeps)
        # mean -> (n_envs,)
        return torch.mean(torch.sum(torch.square(
                self.substep_torques.reshape([self.num_envs * self.num_agents, self.decimation, self.num_dof]) * \
                self.substep_dof_vel.reshape([self.num_envs * self.num_agents, self.decimation, self.num_dof])
            ), dim=-1), dim=-1)

    def _reward_legs_energy_abs(self):
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_alive(self):
        return 1.

    def _reward_lin_cmd(self):
        """ This reward term does not depend on the policy, depends on the command """
        return torch.norm(self.commands[:, :2], dim= 1)

    def _reward_lin_vel_x(self):
        return self.root_states[:, 7]
    
    def _reward_lin_vel_y_abs(self):
        return torch.abs(self.root_states[:, 8])
    
    def _reward_lin_vel_y_square(self):
        return torch.square(self.root_states[:, 8])

    def _reward_lin_pos_y(self):
        return torch.abs((self.root_states[:, :3] - self.env_origins)[:, 1])
    
    def _reward_yaw_abs(self):
        """ Aiming for the robot yaw to be zero (pointing to the positive x-axis) """
        yaw = get_euler_xyz(self.root_states[:, 3:7])[2]
        yaw[yaw > np.pi] -= np.pi * 2 # to range (-pi, pi)
        yaw[yaw < -np.pi] += np.pi * 2 # to range (-pi, pi)
        return torch.abs(yaw)

    def _reward_penetrate_depth(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        penetration_depths = self.terrain.get_penetration_depths(self.volume_sample_points.view(-1, 3)).view(self.num_envs, -1)
        penetration_depths *= torch.norm(self.velocity_sample_points, dim= -1) + 1e-3
        return torch.sum(penetration_depths, dim= -1)

    def _reward_penetrate_volume(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        penetration_mask = self.terrain.get_penetration_mask(self.volume_sample_points.view(-1, 3)).view(self.num_envs, -1)
        penetration_mask *= torch.norm(self.velocity_sample_points, dim= -1) + 1e-3
        return torch.sum(penetration_mask, dim= -1)

    ##### Some helper functions that override parent class attributes #####
    @property
    def all_obs_components(self):
        components = set(self.cfg.env.obs_components)
        if getattr(self.cfg.env, "privileged_obs_components", None):
            components.update(self.cfg.env.privileged_obs_components)
        return components
    
    @property
    def obs_segments(self):
        return self.get_obs_segment_from_components(self.cfg.env.obs_components)
    @property
    def privileged_obs_segments(self):
        components = getattr(
            self.cfg.env,
            "privileged_obs_components",
            None
        )
        if components is None:
            return None
        else:
            return self.get_obs_segment_from_components(components)
    @property
    def num_obs(self):
        """ get this value from self.cfg.env """
        assert "proprioception" in self.cfg.env.obs_components, "missing critical observation component 'proprioception'"
        return self.get_num_obs_from_components(self.cfg.env.obs_components)
    @num_obs.setter
    def num_obs(self, value):
        """ avoid setting self.num_obs """
        pass
    @property
    def num_privileged_obs(self):
        """ get this value from self.cfg.env """
        components = getattr(
            self.cfg.env,
            "privileged_obs_components",
            None
        )
        if components is None:
            return None
        else:
            return self.get_num_obs_from_components(components)
    @num_privileged_obs.setter
    def num_privileged_obs(self, value):
        """ avoid setting self.num_privileged_obs """
        pass

    def check_BarrierTrack_terrain(self):
        if getattr(self.cfg.terrain, "pad_unavailable_info", False):
            return self.cfg.terrain.selected == "BarrierTrack"
        assert self.cfg.terrain.selected == "BarrierTrack", "This implementation is only for BarrierTrack terrain"
        return True
