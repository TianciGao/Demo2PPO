"""
train_ppo.py

目标:
1) 通过 Legacy Remote API 连接 CoppeliaSim (端口=19999)
2) 每次 reset 时, 随机生成红色障碍物(Obstacles) 和一个蓝色方块(目标)
3) 使用 PPO 进行强化学习, 每步计算距离奖励、碰撞惩罚等
4) 提供定制的EvalCallback, 写TensorBoard日志, 并可在D:\program\PPOrobot 下保存模型

使用:
1) 在CoppeliaSim中加载带有UR5机械臂和Lua子脚本的场景(脚本须包含 createScenarioLua, cleanupScenarioLua, collisionCheckLua等函数)
2) 确保仿真启动(▶)
3) 在命令行执行: python train_ppo.py
4) (可选) 在另一命令行执行: tensorboard --logdir="D:\program\PPOrobot\tensorboard_logs"
"""

import os
import sys
import time
import math
import random
import shutil
import numpy as np

# ===========  0) 路径设定 & 日志目录  ===========

ROOT_DIR = "D:\\program\\PPOrobot"  # 你可以根据需要修改
TENSORBOARD_DIR = os.path.join(ROOT_DIR, "tensorboard_logs")
EVAL_LOG_DIR    = os.path.join(ROOT_DIR, "eval_logs")
BEST_MODEL_DIR  = os.path.join(ROOT_DIR, "best_model")

def make_unique_logdir(base_dir):
    """创建一个带时间戳的日志目录, 并清理其中旧PPO_文件."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    t_str = time.strftime("%Y%m%d_%H%M%S")
    rnd   = random.randint(0,99999)
    dir_name = f"run_{t_str}_{rnd:05d}"
    full_path = os.path.join(base_dir, dir_name)

    if os.path.exists(full_path):
        if os.path.isfile(full_path):
            os.remove(full_path)
        else:
            shutil.rmtree(full_path, ignore_errors=True)
    os.makedirs(full_path, exist_ok=True)

    remove_ppo_files_in(full_path)
    return full_path

def remove_ppo_files_in(folder):
    """移除以 PPO_ 开头的文件或文件夹, 避免命名冲突."""
    if not os.path.isdir(folder):
        return
    for fname in os.listdir(folder):
        if fname.startswith("PPO_"):
            p = os.path.join(folder, fname)
            if os.path.isfile(p):
                os.remove(p)
            else:
                shutil.rmtree(p, ignore_errors=True)

# ===========  1) 连接 sim.py & simConst.py  ===========
try:
    import sim       # coppeliaSim legacy remote API
    import simConst  # simConst.py
except:
    print("[Error] cannot import sim.py or simConst.py. Please ensure they are in:", ROOT_DIR)
    sys.exit(-1)

# ===========  2) 定义 UR5Env / GymUr5Env  ===========

import gymnasium as gym
from gymnasium import spaces

class UR5Env:
    """
    低层次的UR5环境封装, 负责:
      - reset_env(): 清理旧障碍, 创建新障碍+目标, 机械臂回到初始位
      - step_env(): 执行6维关节增量动作, 计算距离奖励、碰撞惩罚等
    """
    def __init__(self, port=19999, synchronous=True, max_steps=200):
        self.clientID = -1
        self.port = port
        self.synchronous = synchronous
        self.max_episode_steps = max_steps
        self.current_step = 0

        self.joint_handles = []
        self.tip_handle = -1
        self.target_handle = -1  # 每次随机生成 "Target", 训练时临时获取

        # UR5 六个关节的角度限制, 视实际情况可修改
        self.joint_limits_low  = np.array([-math.pi]*6, dtype=np.float32)
        self.joint_limits_high = np.array([ math.pi]*6, dtype=np.float32)

        # 初始化: 连接CoppeliaSim, 获取对象句柄
        self._connect()
        self._init_handles()

    def _connect(self):
        # 先结束所有可能的连接
        sim.simxFinish(-1)
        # 建立到127.0.0.1:19999端口的连接
        self.clientID = sim.simxStart('127.0.0.1', self.port, True, True, 5000, 5)
        if self.clientID == -1:
            print(f"[UR5Env] fail connect at port={self.port}")
            sys.exit(-1)
        print(f"[UR5Env] connected, clientID={self.clientID}")

        # 同步模式(可选)
        if self.synchronous:
            rc = sim.simxSynchronous(self.clientID, True)
            if rc != sim.simx_return_ok:
                print("[Warn] simxSynchronous => fail => using async mode.")
                self.synchronous = False

    def _init_handles(self):
        # 获取6个关节 handle
        for i in range(1, 7):
            name = f"UR5_joint{i}"
            rc,h = sim.simxGetObjectHandle(self.clientID, name, sim.simx_opmode_blocking)
            if rc==sim.simx_return_ok:
                self.joint_handles.append(h)
            else:
                print(f"[Warn] getObjectHandle({name}) rc={rc}")

        # 获取 tip (UR5末端)
        rc,tipH = sim.simxGetObjectHandle(self.clientID, "tip", sim.simx_opmode_blocking)
        if rc==sim.simx_return_ok:
            self.tip_handle=tipH
            print("[UR5Env] tip handle:", tipH)
        else:
            print("[Warn] cannot find 'tip' alias in scene. rc=", rc)

        self.target_handle=-1  # reset_env时再获取

        # 让tip开始streaming位置
        if self.tip_handle>=0:
            sim.simxGetObjectPosition(self.clientID, self.tip_handle, -1, sim.simx_opmode_streaming)
        time.sleep(0.2)

    def reset_env(self):
        """
        清理旧场景 + 新建随机障碍 / 目标 + 回到初始位姿 + 返回观测
        """
        self.current_step=0

        # 1) cleanup
        self._cleanup_scenario()

        # 2) create random scenario
        self._create_scenario(numObs=3, rMin=0.2, rMax=0.5, hMin=-0.1, hMax=0.1)

        # 3) 获取 "Target" handle
        rc,tH = sim.simxGetObjectHandle(self.clientID, "Target", sim.simx_opmode_blocking)
        if rc==sim.simx_return_ok:
            self.target_handle=tH
        else:
            print("[Warn] cannot find 'Target' shape, rc=", rc)

        # 4) 回到初始姿态
        init_deg = [180, -135, 80, -45, 0, 0]
        init_rad= [math.radians(d) for d in init_deg]
        for jh,ang in zip(self.joint_handles, init_rad):
            sim.simxSetJointPosition(self.clientID, jh, ang, sim.simx_opmode_blocking)
        time.sleep(0.3)

        # 5) streaming
        if self.tip_handle>=0:
            sim.simxGetObjectPosition(self.clientID, self.tip_handle, -1, sim.simx_opmode_streaming)
        if self.target_handle>=0:
            sim.simxGetObjectPosition(self.clientID, self.target_handle, -1, sim.simx_opmode_streaming)
        time.sleep(0.2)

        return self._get_state()

    def step_env(self, action):
        """
        对6个关节施加增量动作, 计算奖励, 检测碰撞, 返回 (obs, reward, done, info)
        """
        scale=0.05  # action范围[-1,1] => 実际关节增量[-0.05, +0.05] (rad)
        increments = scale * np.array(action, dtype=np.float32)

        # 获取当前6关节角
        oldAngles=[]
        for jh in self.joint_handles:
            rc,oa=sim.simxGetJointPosition(self.clientID, jh, sim.simx_opmode_blocking)
            oldAngles.append(oa if rc==sim.simx_return_ok else 0.)

        # 加上增量, 并做clip
        newAngles = [o+a for (o,a) in zip(oldAngles,increments)]
        newAngles_clipped = np.clip(newAngles, self.joint_limits_low, self.joint_limits_high)

        # 下发关节命令
        for jh, na in zip(self.joint_handles, newAngles_clipped):
            sim.simxSetJointTargetPosition(self.clientID, jh, na, sim.simx_opmode_oneshot)

        if self.synchronous:
            sim.simxSynchronousTrigger(self.clientID)
        time.sleep(0.1)

        # 获取观测
        obs = self._get_state()
        tip = obs[6:9]
        tgt = obs[9:12]
        dist= float(np.linalg.norm(tip - tgt))

        # 计算奖励 (距离惩罚 + 小步惩罚 + 临近奖励等)
        reward = -dist - 0.01
        if dist<0.1:
            reward +=1
        if dist<0.05:
            reward +=2

        done=False
        collision_flag=False
        success_flag=False

        # 若距离足够近, 视为成功
        if dist<0.02:
            reward +=5
            success_flag=True
            done=True

        # 关节是否超限
        out_of_limit= not np.allclose(newAngles, newAngles_clipped, atol=1e-7)
        if out_of_limit:
            reward-=15
            done=True

        # 碰撞检测
        collision_flag = (self._collisionCheck()==1)
        if collision_flag:
            reward-=10
            done=True

        # 步数限制
        self.current_step +=1
        if self.current_step>=self.max_episode_steps and not done:
            done=True

        info={
            "collision": collision_flag,
            "success": success_flag,
            "out_of_limit": out_of_limit
        }
        return obs, float(reward), done, info

    def _collisionCheck(self):
        """
        调用Lua子脚本: collisionCheckLua
        返回1表示碰撞, 0表示无碰撞
        """
        ret,outInts,outFloats,outStrings,outBuffer = sim.simxCallScriptFunction(
            self.clientID,
            "UR5",  # lua脚本所在对象别名
            simConst.sim_scripttype_childscript,
            "collisionCheckLua",
            [],[],[], "",
            sim.simx_opmode_blocking
        )
        if ret==sim.simx_return_ok and len(outInts)>0:
            return outInts[0]
        return 0

    def _cleanup_scenario(self):
        """
        调用Lua: cleanupScenarioLua => 删除上一回合生成的障碍和目标
        """
        sim.simxCallScriptFunction(
            self.clientID,
            "UR5",
            simConst.sim_scripttype_childscript,
            "cleanupScenarioLua",
            [],[],[], "",
            sim.simx_opmode_blocking
        )

    def _create_scenario(self, numObs=3, rMin=0.2, rMax=0.5, hMin=-0.1, hMax=0.1):
        """
        调用Lua: createScenarioLua => 生成随机的红色障碍 & 蓝色方块
        """
        inInts = [numObs]
        inFloats= [rMin, rMax, hMin, hMax]
        sim.simxCallScriptFunction(
            self.clientID,
            "UR5",
            simConst.sim_scripttype_childscript,
            "createScenarioLua",
            inInts,
            inFloats,
            [],
            "",
            sim.simx_opmode_blocking
        )

    def _get_position_with_retry(self, handle, tries=10):
        """
        多次尝试获取位置, 兼容streaming延迟
        """
        pos=[0,0,0]
        for i in range(tries):
            rc,pos_ = sim.simxGetObjectPosition(
                self.clientID, handle, -1, sim.simx_opmode_buffer
            )
            if rc==sim.simx_return_ok:
                pos=pos_
                break
            else:
                time.sleep(0.05)
        return pos

    def _get_state(self):
        """
        返回 [joint1..6, tip_x, tip_y, tip_z, tgt_x, tgt_y, tgt_z]
        """
        st=[]
        for jh in self.joint_handles:
            rc,ang = sim.simxGetJointPosition(self.clientID, jh, sim.simx_opmode_blocking)
            st.append(ang if rc==sim.simx_return_ok else 0.)

        tipPos=[0,0,0]
        if self.tip_handle>=0:
            tipPos=self._get_position_with_retry(self.tip_handle)
        tgtPos=[0,0,0]
        if self.target_handle>=0:
            tgtPos=self._get_position_with_retry(self.target_handle)

        st.extend(tipPos)
        st.extend(tgtPos)
        return np.array(st, dtype=np.float32)

    def close(self):
        print("[UR5Env] closed.")


class GymUr5Env(gym.Env):
    """
    使用 gymnasium Env API 再包装一层, 方便后续 stable-baselines3 调用.
    """
    def __init__(self, port=19999, synchronous=True, max_steps=200):
        super().__init__()
        self.ur5 = UR5Env(port=port, synchronous=synchronous, max_steps=max_steps)

        # 动作空间: [-1, +1]^6 => 6关节增量
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        # 状态空间: 12维 (6关节 + tip_xyz + target_xyz)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.ur5.reset_env()
        info={}
        return obs, info

    def step(self, action):
        obs_next, reward, done, info = self.ur5.step_env(action)
        truncated=False
        return obs_next, reward, done, truncated, info

    def close(self):
        self.ur5.close()

# =========== 3) 自定义EvalCallback ===========

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

class MyEvalCallback(EvalCallback):
    """
    自定义评估回调: 每2000步评估1次, 并记录 success_rate, collision_rate 等.
    """
    def __init__(self, eval_env, n_eval_episodes=5, eval_freq=2000,
                 best_model_save_path=None, log_path=None, verbose=1):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            verbose=verbose
        )
        self.success_count=0
        self.collision_count=0

    def _evaluate_policy(self):
        self.success_count=0
        self.collision_count=0
        episode_rewards=[]
        episode_lengths=[]

        for _ in range(self.n_eval_episodes):
            obs,_ = self.eval_env.reset()
            done=False
            truncated=False
            ep_reward=0.
            ep_length=0
            while not (done or truncated):
                action,_ = self.model.predict(obs, deterministic=self.deterministic)
                obs,reward,done,truncated,info = self.eval_env.step(action)
                ep_reward+=reward
                ep_length+=1

                if info.get("success",False):
                    self.success_count+=1
                if info.get("collision",False):
                    self.collision_count+=1

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

        mean_reward = float(np.mean(episode_rewards))
        std_reward  = float(np.std(episode_rewards))
        mean_ep_len = float(np.mean(episode_lengths))

        success_rate= (self.success_count / self.n_eval_episodes)*100
        collision_rate= (self.collision_count / self.n_eval_episodes)*100

        if self.verbose>0:
            print(f"[Eval] success={self.success_count}, collision={self.collision_count}")
            print(f"[Eval] success_rate={success_rate:.1f}%, collision_rate={collision_rate:.1f}%")
            print(f"[Eval] mean_reward={mean_reward:.2f}, mean_length={mean_ep_len:.2f}")

        if self.model.logger is not None:
            self.model.logger.record("eval/success_rate", success_rate)
            self.model.logger.record("eval/collision_rate", collision_rate)
            self.model.logger.record("eval/mean_reward", mean_reward)
            self.model.logger.record("eval/mean_ep_len", mean_ep_len)
            self.model.logger.dump(self.model.num_timesteps)

        return mean_reward, std_reward

# =========== 4) Main: 训练与测试 ===========

def main():
    # 1) 日志/模型目录
    os.makedirs(EVAL_LOG_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    tb_dir = make_unique_logdir(TENSORBOARD_DIR)

    # 2) 构建env, 先检查其有效性
    env = GymUr5Env(port=19999, synchronous=True, max_steps=200)
    check_env(env, warn=True)
    eval_env = GymUr5Env(port=19999, synchronous=True, max_steps=200)

    # 3) 自定义评估回调
    eval_callback = MyEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=5,
        eval_freq=2000,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=EVAL_LOG_DIR,
        verbose=1
    )

    # 4) 创建PPO并开始训练
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log=tb_dir
    )
    total_timesteps=50000
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save("ppo_ur5_model")
    print("[Train] PPO model saved => ppo_ur5_model")

    # 5) 测试一下
    obs,_ = env.reset()
    done=False
    truncated=False
    total_reward=0.
    steps=0
    while not (done or truncated):
        action,_= model.predict(obs, deterministic=True)
        obs,reward,done,truncated,info = env.step(action)
        total_reward+=reward
        steps+=1
        time.sleep(0.1)

    print(f"[TestOne] done steps={steps}, total_reward={total_reward:.2f}, "
          f"success={info.get('success',False)}, collision={info.get('collision',False)}")

    env.close()
    eval_env.close()

if __name__=="__main__":
    main()
