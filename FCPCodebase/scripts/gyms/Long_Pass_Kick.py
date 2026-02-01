# scripts/gyms/Long_Pass_Kick.py

from agent.Base_Agent import Base_Agent as Agent
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from world.commons.Draw import Draw
from time import sleep
import gym
import numpy as np
import os
import time
import socket



class EpisodeStatsCallback(BaseCallback):
    """Print episode stats (dist/height) from env info when done."""
    def __init__(self, print_every=10, verbose=0):
        super().__init__(verbose)
        self.print_every = int(print_every)
        self.ep = 0
    def _on_step(self) -> bool:
        # infos is a list (vec env). dones is np array
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        for d, info in zip(dones, infos):
            if d:
                self.ep += 1
                if info is None:
                    continue
                dist = info.get("dist", None)
                height = info.get("height", None)
                abs_y = info.get("abs_y", None)
                if self.ep % self.print_every == 0:
                    if dist is None or height is None or abs_y is None:
                        continue
                    hit = int(info.get("hit", 0))
                    stage = info.get("stage", None)
                    goal = info.get("goal_dist", None)
                    vxy = info.get("max_vxy", None)
                    if stage is None or goal is None or vxy is None:
                        print(f"[EP {self.ep}] dist={dist:.2f}m  height={height:.2f}m  |y|={abs_y:.2f}m")
                    else:
                        print(f"[EP {self.ep}] hit={hit} stage={stage} goal={goal:.0f}m  dist={dist:.2f}m  h={height:.2f}m  vxy={vxy:.2f}  |y|={abs_y:.2f}m")
        return True



class StepHeartbeatCallback(BaseCallback):
    """Print a heartbeat every N env steps so you know training is progressing."""
    def __init__(self, every_steps=256, verbose=0):
        super().__init__(verbose)
        self.every_steps = int(every_steps)
    def _on_step(self) -> bool:
        if self.num_timesteps % self.every_steps == 0:
            print(f"[STEP] num_timesteps={self.num_timesteps}")
        return True


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
        # Prevent silent hangs if server stops responding
        try:
            self.player.scom.socket.settimeout(5.0)
        except Exception:
            pass

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

        # Action = [delta_residual, 6 joint residuals]
        # 6 residuals only touch key joints of the right-leg kick (robot type dependent).
        # This greatly reduces exploration space and makes training more stable.
        # Joint IDs follow the slot XML ids (see behaviors/slot/r*/Kick_Motion.xml).
        # Tuned joints: [3]=R yaw/pitch, [5]=L roll, [6]=L pitch, [7]=R pitch, [9]=R knee, [11]=R foot pitch
        self.tuned_joint_ids = np.array([3, 5, 6, 7, 9, 11], dtype=np.int32)
        self.n_tuned = int(self.tuned_joint_ids.size)

        # We always rewrite the slot as (slice(0,22), full_angles_22) like Get_Up does.
        MAX = np.finfo(np.float32).max
        self.action_space = gym.spaces.Box(
            low=np.full(1 + self.n_tuned, -MAX, np.float32),
            high=np.full(1 + self.n_tuned,  MAX, np.float32),
            dtype=np.float32
        )

        self.step_counter = 0

        # Kick scenario config
        self.robot_beam = (-13.0, 0.0)   # robot position
        self.robot_ori  = 0.0            # face +x
        self.ball_rel_base = (0.205, -0.11)  # nominal placement near right foot
        self.ball_rel      = list(self.ball_rel_base)  # actual placement may be jittered per reset
        self.ball_z        = 0.042           # ball radius height

        # --- hit-rate helpers (auto-calibrate ball side) ---
        self._ball_y_sign  = -1.0            # start by placing ball at y<0 (right-foot side)
        self._miss_streak  = 0
        self._last_dist    = 0.0
        self._last_hit     = False

        # --- curriculum / scaling ---
        self.global_steps      = 0           # counts env.step calls (RL steps)
        self.episode_counter   = 0
        self.deg_base, self.deg_max = 2.0, 10.0
        self.ms_base,  self.ms_max  = 25.0, 60.0

        self.max_follow_steps = 80      # observe ball travel after kick
        # observe ball flight (legacy comment)

        self._ball_start = None


    def _sync(self):
        """Run a single simulation step (with safety timeout)"""
        r = self.player.world.robot
        self.player.scom.commit_and_send(r.get_command())
        try:
            self.player.scom.receive()
        except socket.timeout as e:
            raise RuntimeError("Server_Comm.receive() timed out") from e
        except Exception as e:
            # Let the worker crash so the outer training loop can restart servers
            raise


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


    
    # ---------------- curriculum & shaping helpers ----------------
    def _progress(self) -> float:
        """0..1 training progress based on how many RL steps we have run."""
        return float(min(1.0, self.global_steps / 200_000.0))

    def _curriculum_params(self):
        """Returns (stage, goal_dist, min_height, jitter_xy)."""
        # stage boundaries (in RL steps)
        gs = self.global_steps
        if gs < 30_000:
            return 0, 6.0, 0.10, (0.000, 0.000)
        elif gs < 120_000:
            return 1, 12.0, 0.20, (0.010, 0.010)
        elif gs < 250_000:
            return 2, 20.0, 0.30, (0.015, 0.015)
        else:
            return 3, 30.0, 0.45, (0.020, 0.020)

    def _action_scales(self):
        """Scale factors for (delta_ms, tuned_deg) with a smooth ramp."""
        p = self._progress()
        delta_scale = self.ms_base + p * (self.ms_max - self.ms_base)
        deg_scale   = self.deg_base + p * (self.deg_max - self.deg_base)
        return float(delta_scale), float(deg_scale)

    def _sample_ball_rel(self):
        """Ball placement: auto-calibrated side + curriculum jitter."""
        stage, _, _, (jx, jy) = self._curriculum_params()
        bx, by = self.ball_rel_base
        # small randomization for robustness (curriculum controls how big)
        rx = (np.random.rand() * 2.0 - 1.0) * jx
        ry = (np.random.rand() * 2.0 - 1.0) * jy
        # y sign auto-calibration
        y = abs(by) * float(self._ball_y_sign) + ry
        x = bx + rx
        return (float(x), float(y))

    def _update_ball_calibration(self, dist: float, hit: bool):
        """Flip ball side if we keep missing (helps kick-foot mismatch)."""
        if hit:
            self._miss_streak = 0
            return
        # miss
        self._miss_streak += 1
        if self._miss_streak >= 12:
            self._ball_y_sign *= -1.0
            self._miss_streak = 0
            print(f"[CALIB] Too many misses -> flipping ball_rel y sign to {self._ball_y_sign:+.0f}")

    def reset(self):
        self.step_counter = 0
        # --- safety: initialize ball start even if reset exits early ---
        try:
            self._ball_start = self.player.world.ball_abs_pos.copy()
        except Exception:
            self._ball_start = None
        self.current_slot = 0
        self._restore_original_kick_motion()

        # ✅ 用官方的 beam 方式（2D）
        self.player.scom.commit_beam(self.robot_beam, self.robot_ori)

        # 走几步让世界稳定
        for _ in range(3):
            self.player.behavior.execute("Zero_Bent_Knees")
            self._sync()

        # ✅ 放球到脚前（cheat move ball）
        # curriculum jitter + auto-calibrated side
        self.ball_rel = list(self._sample_ball_rel())
        bx = self.robot_beam[0] + self.ball_rel[0]
        by = self.robot_beam[1] + self.ball_rel[1]
        self.player.scom.unofficial_move_ball((bx, by, self.ball_z), (0, 0, 0))

        for _ in range(3):
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
        # delta_res affects milliseconds; tuned_res affects degrees
        # We clip to [-1,1] and then scale progressively (curriculum).
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        delta_scale, deg_scale = self._action_scales()
        delta_res = float(action[0]) * delta_scale
        tuned_res = action[1:].astype(np.float32) * deg_scale
        self._last_action_l2 = float(np.linalg.norm(action[1:]))
        self.global_steps += 1

        # Read original slot
        orig_delta_ms, orig_indices, orig_angles_sparse = self.original_slots[self.current_slot]
        orig_full_22 = self._get_22_angles(orig_indices, orig_angles_sparse)

        # Apply residual
        new_delta_ms = np.clip(orig_delta_ms + delta_res, 60.0, 400.0)
        new_full_22 = orig_full_22.copy()
        new_full_22[self.tuned_joint_ids] += tuned_res

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
        max_vxy = 0.0
        max_vx = -1e9
        air_steps = 0

        for _ in range(self.max_follow_steps):
            self.player.behavior.execute("Zero_Bent_Knees")
            self._sync()

            b = self.player.world.ball_abs_pos
            v = self.player.world.ball_abs_vel
            vxy = float(np.linalg.norm(v[:2]))
            max_vxy = max(max_vxy, vxy)
            max_vx = max(max_vx, float(v[0]))
            if b[2] > 0.15:
                air_steps += 1
            max_x = max(max_x, b[0])
            max_z = max(max_z, b[2])
            max_abs_y = max(max_abs_y, abs(b[1]))

            # stop early if ball almost stopped
            if np.linalg.norm(self.player.world.ball_abs_vel[:2]) < 0.05:
                break

        # Compute metrics
        # --- safety: ball_start may be None if reset failed to move/beam ---
        if self._ball_start is None:
            try:
                self._ball_start = self.player.world.ball_abs_pos.copy()
            except Exception:
                self._ball_start = np.array([0.0,0.0,0.0], dtype=np.float32)
        start_x = float(self._ball_start[0])
        dist = float(max_x - start_x)
        height = float(max_z)

        # -------- shaped terminal reward (A~E) --------

        stage, goal_dist, min_h, _ = self._curriculum_params()

        # hit detection (踢到球): either it traveled or it gained speed
        hit = (dist >= 0.50) or (max_vxy >= 0.25)

        # Curriculum-aware bonuses
        hit_bonus = 2.5 if hit else -4.0
        miss_penalty = -6.0 if (dist < 0.50 and max_vxy < 0.20) else 0.0  # B: 踢空强惩罚

        # Core objectives: distance + loft (high ball)
        forward_dist = max(0.0, dist)
        backward_pen = -2.0 * max(0.0, -dist)

        # C: reward 更密集（多项指标组合，而不是只看最后距离）
        speed_term = 0.25 * max_vxy + 0.10 * max(0.0, max_vx)
        loft_term  = 4.0 * max(0.0, height - min_h)        # encourage higher than curriculum min
        air_bonus  = 0.30 * min(1.0, air_steps / 15.0)     # time in air

        # D: 课程式训练 —— 达到阶段目标距离会给额外奖励
        goal_bonus = 3.0 * max(0.0, forward_dist - goal_dist) / max(1.0, goal_dist)

        # Small regularizers
        align_pen  = 0.20 * max_abs_y                      # keep y deviation small
        act_pen    = 0.05 * getattr(self, "_last_action_l2", 0.0)  # discourage extreme actions early

        reward = (
            1.0 * forward_dist
            + hit_bonus + miss_penalty + backward_pen
            + speed_term + loft_term + air_bonus + goal_bonus
            - align_pen - act_pen
        )

        # If it's not lofting yet, down-weight distance so it doesn't learn a flat ground pass only
        if height < min_h:
            reward *= 0.40

        # Update auto-calibration for ball placement (A)
        self._update_ball_calibration(dist, hit)

        self._last_dist = dist
        self._last_hit = hit
        info = {
            "dist": dist,
            "height": height,
            "abs_y": max_abs_y,
            "hit": int(hit),
            "max_vxy": float(max_vxy),
            "stage": int(stage),
            "goal_dist": float(goal_dist),
            "min_h": float(min_h),
            "ball_y_sign": float(self._ball_y_sign),
        }
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
        """
        Robust training loop:
        - Runs training in chunks and checkpoints to latest.zip.
        - If rcssserver3d crashes (socket closed -> worker dies -> EOFError),
          we save latest, kill leftover servers, restart and resume.
        - Uses SubprocVecEnv (so a worker exit() won't kill the main process).
        """
        import subprocess
        import threading
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback

        # ---- stability-first settings ----
        n_envs = 2  # MUST be >=2 so scripts/commons/Server.py uses rcssserver3d (not simspark)
        n_steps_per_env = 128
        minibatch_size = 128
        total_steps = 300_000

        # Chunking = less time per server session (reduces "long-run random crash" chance)
        chunk_steps = 10_000
        save_every_steps = 2_000  # checkpoints

        # Where to save
        # Run_Utils passes args as a dict. For Train it is usually empty.
        model_path = None
        if isinstance(args, dict):
            model_path = args.get("folder_dir", None)
        if not model_path:
            folder_name = f"Long_Pass_Kick_R{self.robot_type}_HighLobPass_v1"
            model_path = f"./scripts/gyms/logs/{folder_name}/"
        os.makedirs(model_path, exist_ok=True)
        print("Model path:", model_path)
        latest_model = os.path.join(model_path, "latest.zip")
        learning_rate = 3e-4

        def init_env(rank):
            def _init():
                draw_host = getattr(self.script.args, "d", self.ip)
                env = Long_Pass_Kick(
                    self.ip,
                    self.server_p + rank,
                    self.monitor_p_1000 + rank,
                    self.robot_type,
                    False,
                    draw_host=draw_host,
                )
                return env
            return _init

        def pre_kill_leftovers():
            """
            Kill leftover rcssserver3d/simspark processes that may keep ports busy.
            Avoids Server.py blocking on interactive 'Enter kill...' prompt.
            """
            for i in range(n_envs):
                ap = self.server_p + i
                sp = self.monitor_p_1000 + i
                # Best-effort: only matches servers started with these ports
                subprocess.run(["pkill", "-9", "-f", f"rcssserver3d --agent-port {ap} --server-port {sp}"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["pkill", "-9", "-f", f"simspark --agent-port {ap} --server-port {sp}"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def safe_close_vecenv(env, timeout_sec=2.0):
            """
            SubprocVecEnv.close() can hang if workers are blocked.
            Close in a daemon thread; if it times out, force-kill workers.
            """
            if env is None:
                return

            # Try graceful close in background (non-blocking)
            def _close():
                try:
                    env.close()
                except Exception as e:
                    print("[WARN] env.close() raised:", repr(e))

            t = threading.Thread(target=_close, daemon=True)
            t.start()
            t.join(timeout=timeout_sec)

            if t.is_alive():
                # Force kill worker processes
                try:
                    procs = getattr(env, "processes", None)
                    if procs:
                        for p in procs:
                            try:
                                if p.is_alive():
                                    p.terminate()
                            except Exception:
                                pass
                        for p in procs:
                            try:
                                p.join(timeout=1)
                            except Exception:
                                pass
                        for p in procs:
                            try:
                                if p.is_alive():
                                    p.kill()
                            except Exception:
                                pass
                    print("[WARN] env.close() timed out -> workers killed.")
                except Exception as e:
                    print("[WARN] force-kill env workers failed:", repr(e))

        # ---- resume or create model ----
        model = None
        steps_done = 0
        restarts = 0
        max_restarts = 999999  # basically infinite

        while steps_done < total_steps and restarts < max_restarts:
            env = None
            servers = None

            # Always kill leftovers BEFORE starting servers (prevents input() prompt)
            pre_kill_leftovers()

            try:
                servers = Server(self.server_p, self.monitor_p_1000, n_envs)
                time.sleep(0.5)

                env = SubprocVecEnv([init_env(i) for i in range(n_envs)])

                if model is None:
                    if os.path.isfile(latest_model):
                        print("[RESUME] Loading", latest_model)
                        model = PPO.load(latest_model, env=env, device="cpu")
                    else:
                        model = PPO(
                            "MlpPolicy",
                            env=env,
                            verbose=1,
                            n_steps=n_steps_per_env,
                            batch_size=minibatch_size,
                            learning_rate=learning_rate,
                            device="cpu",
                        )
                else:
                    model.set_env(env)

                # callbacks
                callbacks = [
                    CheckpointCallback(save_freq=save_every_steps, save_path=model_path, name_prefix="ckpt"),
                    StepHeartbeatCallback(every_steps=256),
                    EpisodeStatsCallback(print_every=1),
                ]

                remaining = total_steps - steps_done
                this_chunk = min(chunk_steps, remaining)
                print(f"[TRAIN] steps_done={steps_done}  training_chunk={this_chunk}  restarts={restarts}")

                model.learn(total_timesteps=this_chunk, reset_num_timesteps=False, callback=callbacks)

                steps_done += this_chunk

                # Always save "latest" after a chunk
                model.save(latest_model)
                print(f"[SAVE] saved latest to {latest_model}")

                # Restart server each chunk to reduce long-run crashes
                restarts += 1

            except Exception as e:
                # Crash path: save latest and retry
                print("[WARN] Server crashed:", repr(e))
                try:
                    if model is not None:
                        model.save(latest_model)
                        print(f"[SAVE] saved latest to {latest_model}")
                except Exception as se:
                    print("[WARN] saving latest failed:", repr(se))

                restarts += 1
                # small cool down
                time.sleep(1.0)

            finally:
                # Clean up env/server, but NEVER block here
                safe_close_vecenv(env, timeout_sec=2.0)
                try:
                    if servers is not None:
                        servers.kill()
                except Exception as ke:
                    print("[WARN] servers.kill() failed:", repr(ke))

                # extra cooldown to avoid port race
                time.sleep(0.5)

        print("[DONE] Training loop finished. steps_done=", steps_done)


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

