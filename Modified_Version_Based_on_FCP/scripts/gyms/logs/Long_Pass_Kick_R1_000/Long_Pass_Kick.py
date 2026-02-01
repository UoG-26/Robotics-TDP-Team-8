# scripts/gyms/Long_Pass_Kick.py

from agent.Base_Agent import Base_Agent as Agent
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from world.commons.Draw import Draw
from time import sleep
import gym
import numpy as np
import os


class Long_Pass_Kick(gym.Env):
    """
    Objective:
    Optimize the existing slot behavior 'Kick_Motion' to create a high long pass (30m+).
    Similar approach to Get_Up: optimize keyframes (slots) instead of controlling every frame.
    """

    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw, draw_host=None) -> None:
        self.robot_type = r_type

        # IMPORTANT:
        # - ip is server host
        # - draw_host is where RoboViz listens (Windows IP / WSL gateway)
        if draw_host is None:
            draw_host = ip

        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw
        # (Base_Agent has extra optional args, so passing these 8 is safe like Basic_Run/Get_Up)
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw, draw_host=draw_host)

        # Backup original Kick_Motion slots (from Slot_Engine)
        self.beh_name = "Kick_Motion"
        slots = self.player.behavior.slot_engine.behaviors[self.beh_name]

        self.original_slots = []
        for delta_ms, indices, angles in slots:
            self.original_slots.append((delta_ms, indices, np.array(angles, dtype=np.float32)))

        self.n_slots = len(self.original_slots)
        self.current_slot = 0

        # Observation = one-hot slot index (same pattern as Get_Up)
        self.obs = np.identity(self.n_slots, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.n_slots, np.float32),
            high=np.ones(self.n_slots, np.float32),
            dtype=np.float32
        )

        # Action = [delta_residual, 22 joint residuals]
        # We always rewrite the slot as (slice(0,22), full_angles_22) like Get_Up does.
        MAX = np.finfo(np.float32).max
        self.action_space = gym.spaces.Box(
            low=np.full(1 + 22, -MAX, np.float32),
            high=np.full(1 + 22,  MAX, np.float32),
            dtype=np.float32
        )

        self.step_counter = 0

        # Kick scenario config
        self.robot_beam = (-13.0, 0.0)   # robot position
        self.robot_ori  = 0.0            # face +x
        self.ball_rel   = (0.205, -0.11) # ball in right-foot kick area (based on Basic_Kick geometry)
        self.ball_z     = 0.042          # ball radius height
        self.max_follow_steps = 180      # observe ball flight for ~3.6s (180*0.02)

        self._ball_start = None


    def _sync(self):
        """Run a single simulation step"""
        r = self.player.world.robot
        self.player.scom.commit_and_send(r.get_command())
        self.player.scom.receive()


    @staticmethod
    def _get_22_angles(indices, angles):
        """Convert sparse indices+angles into a full 22 vector"""
        full = np.zeros(22, np.float32)
        full[indices] = angles
        return full


    def _restore_original_kick_motion(self):
        slots = []
        for delta_ms, indices, angles in self.original_slots:
            slots.append((delta_ms, indices, angles.copy()))
        self.player.behavior.slot_engine.behaviors[self.beh_name] = slots


    def reset(self):
        self.step_counter = 0
        self.current_slot = 0
        self._restore_original_kick_motion()

        # ✅ 用官方的 beam 方式（2D）
        self.player.scom.commit_beam(self.robot_beam, self.robot_ori)

        # 走几步让世界稳定
        for _ in range(10):
            self.player.behavior.execute("Zero_Bent_Knees")
            self._sync()

        # ✅ 放球到脚前（cheat move ball）
        bx = self.robot_beam[0] + self.ball_rel[0]
        by = self.robot_beam[1] + self.ball_rel[1]
        self.player.scom.unofficial_move_ball((bx, by, self.ball_z), (0, 0, 0))

        for _ in range(10):
            self.player.behavior.execute("Zero_Bent_Knees")
            self._sync()

        self._ball_start = self.player.world.ball_abs_pos.copy()
        return self.obs[self.current_slot]


    def step(self, action):
        """
        Each step optimizes ONE slot of Kick_Motion.
        When all slots are set -> execute Kick_Motion once and measure ball flight.
        """

        # Scale actions (important)
        # delta_res affects milliseconds; angle_res affects degrees
        delta_res = float(action[0]) * 40.0          # ~ +/-40ms
        angle_res = action[1:].astype(np.float32) * 3.0   # ✅先稳住
 # ~ +/-8deg residuals

        # Read original slot
        orig_delta_ms, orig_indices, orig_angles_sparse = self.original_slots[self.current_slot]
        orig_full_22 = self._get_22_angles(orig_indices, orig_angles_sparse)

        # Apply residual
        new_delta_ms = np.clip(orig_delta_ms + delta_res, 60.0, 400.0)
        new_full_22 = orig_full_22 + angle_res

        # Write back to slot engine (force full 22 angles)
        slots = self.player.behavior.slot_engine.behaviors[self.beh_name]
        slots[self.current_slot] = (new_delta_ms, slice(0, 22), new_full_22)
        self.player.behavior.slot_engine.behaviors[self.beh_name] = slots

        self.current_slot += 1

        # Not finished setting slots yet
        if self.current_slot < self.n_slots:
            return self.obs[self.current_slot], 0.0, False, {}

        # ------------------------------------------------------------
        # Terminal: run the optimized Kick_Motion + measure the flight
        # ------------------------------------------------------------
        # Execute kick motion
        finished = False
        first = True
        while not finished:
            finished = self.player.behavior.execute(self.beh_name) if first else self.player.behavior.execute(self.beh_name)
            first = False
            self._sync()
            self.step_counter += 1
            # safety break
            if self.step_counter > 200:
                break

        # Follow-through: watch ball travel
        max_x = -1e9
        max_z = -1e9
        max_abs_y = 0.0

        for _ in range(self.max_follow_steps):
            self.player.behavior.execute("Zero_Bent_Knees")
            self._sync()

            b = self.player.world.ball_abs_pos
            max_x = max(max_x, b[0])
            max_z = max(max_z, b[2])
            max_abs_y = max(max_abs_y, abs(b[1]))

            # stop early if ball almost stopped
            if np.linalg.norm(self.player.world.ball_abs_vel[:2]) < 0.05:
                break

        # Compute metrics
        start_x = float(self._ball_start[0])
        dist = float(max_x - start_x)
        height = float(max_z)

        # Encourage high ball:
        # - dist matters most
        # - height must be "real loft" (>= ~0.45m) otherwise penalize
        reward = dist + 6.0 * max(0.0, height - 0.25) - 0.2 * max_abs_y

        if height < 0.45:
            reward *= 0.25  # basically says: don't accept ground passes

        # Penalize falling
        r = self.player.world.robot
        if r.cheat_abs_pos[2] < 0.30:
            reward -= 5.0

        info = {"dist": dist, "height": height, "abs_y": max_abs_y}
        obs_dummy = self.obs[0]

        return obs_dummy, reward, True, info


    def render(self, mode="human", close=False):
        return


    def close(self):
        Draw.clear_all()
        self.player.terminate()



class Train(Train_Base):
    """
    Training/testing wrapper (same style as Basic_Run / Get_Up)
    """
    def __init__(self, script) -> None:
        super().__init__(script)


    def train(self, args):
        n_envs = 4               # 先别开10个，4个最稳
        n_steps_per_env = 128
        minibatch_size = 256     # 128*4 = 512，256是因子，不会再出现truncated batch警告
        total_steps = 2_000_000  # 先跑小一点验证稳定
        learning_rate = 3e-4


        folder_name = f"Long_Pass_Kick_R{self.robot_type}"
        model_path = f"./scripts/gyms/logs/{folder_name}/"
        print("Model path:", model_path)

        draw_host = getattr(args, "d", self.ip)

        def init_env(i_env):
            def thunk():
                return Long_Pass_Kick(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env,
                                     self.robot_type, False, draw_host=draw_host)
            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)  # +1 eval

        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        eval_env = SubprocVecEnv([init_env(n_envs)])

        try:
            if "model_file" in args:  # retrain
                model = PPO.load(args["model_file"], env=env, device="cpu",
                                 n_envs=n_envs, n_steps=n_steps_per_env,
                                 batch_size=minibatch_size, learning_rate=learning_rate)
            else:
                model = PPO("MlpPolicy", env=env, verbose=1,
                            n_steps=n_steps_per_env, batch_size=minibatch_size,
                            learning_rate=learning_rate, device="cpu")

            self.learn_model(model, total_steps, model_path,
                             eval_env=eval_env,
                             eval_freq=n_steps_per_env * 20,
                             save_freq=n_steps_per_env * 200,
                             backup_env_file=__file__)
        except KeyboardInterrupt:
            sleep(1)
            print("\nctrl+c pressed, aborting...\n")
            servers.kill()
            return

        env.close()
        eval_env.close()
        servers.kill()


    def test(self, args):
        # Uses different server and monitor ports
        server = Server(self.server_p - 1, self.monitor_p, 1)

        draw_host = getattr(self.script.args, "d", self.ip)
        env = Long_Pass_Kick(self.ip, self.server_p - 1, self.monitor_p,
                             self.robot_type, True, draw_host=draw_host)

        model = PPO.load(args["model_file"], env=env)

        try:
            # Export pkl so you can deploy as a custom behavior if you want (same as Basic_Run)
            self.export_model(args["model_file"], args["model_file"] + ".pkl", False)

            # Run interactive test loop (same style)
            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])
        except KeyboardInterrupt:
            print()

        env.close()
        server.kill()

