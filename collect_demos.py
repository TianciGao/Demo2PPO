# collect_demos.py

import os
import numpy as np
import time
import math
import sim
import simConst

# 导入你现有的GymUr5Env (含 reset(), step() 等)
from ppo2 import GymUr5Env


def plan_path_for_scene(env, maxTime=3.0, maxRetries=3, verbose=True):
    """
    调用Lua: computePathOMPLLua, 获取OMPL规划轨迹.
    若出现 ret!=sim.simx_return_ok 或 path过短, 多次重试.
    """

    best_path = []
    for attempt in range(maxRetries):
        obs, _ = env.reset()  # => cleanupScenarioLua + createScenarioLua
        start_joints = obs[:6]

        if verbose:
            print(f"[plan_path_for_scene] attempt={attempt + 1}, start_joints={start_joints}")

        ret, outInts, outFloats, outStrings, outBuffer = sim.simxCallScriptFunction(
            env.ur5.clientID,
            "UR5",
            simConst.sim_scripttype_childscript,
            "computePathOMPLLua",
            [],
            list(start_joints),
            [],
            "",
            sim.simx_opmode_blocking
        )
        if ret != sim.simx_return_ok:
            # OMPL脚本调用失败, 可能是 table contains non-numbers
            if verbose:
                print(f"[Error] computePathOMPLLua call fail, ret={ret}")
            continue  # 重试

        pathData = outFloats
        if len(pathData) < 6:
            if verbose:
                print("[Warn] OMPL returned empty path!")
            continue

        # 解析
        n_waypoints = len(pathData) // 6
        path = []
        for i in range(n_waypoints):
            q = pathData[6 * i: 6 * i + 6]
            path.append(q)

        if verbose:
            print(f"[plan_path_for_scene] => path attempt={attempt + 1}, #waypoints={len(path)}")

        if len(path) > 2:
            best_path = path
            break
        else:
            best_path = path

    return best_path


def build_state_from_joints(env, joint_angles, real_tip_target=False):
    """
    构建与PPO相同的观测 s=[joint1..6, tipXYZ, targetXYZ].
    real_tip_target=True时, 强制把机械臂跳到joint_angles再读tip/target.
    """
    st = list(joint_angles)
    if not real_tip_target:
        st.extend([0, 0, 0])
        st.extend([0, 0, 0])
    else:
        # 执行关节, 读取真实tip/target
        env.ur5._set_joint_position_instant(joint_angles)
        tipPos = env.ur5._get_position(env.ur5.tip_handle)
        tgtPos = [0, 0, 0]
        if env.ur5.target_handle >= 0:
            tgtPos = env.ur5._get_position(env.ur5.target_handle)
        st.extend(tipPos)
        st.extend(tgtPos)
    return np.array(st, dtype=np.float32)


def collect_demonstrations(num_episodes=20,
                           out_file="demo_data.npz",
                           real_tip_target=False,
                           maxTime=3.0,
                           maxRetries=3,
                           verbose=True):
    """
    多次采集OMPL轨迹 => (s,a).
    如果多次出现 table contains non-numbers 或 ret=8,
    可以直接终止并提示在Lua端检查 collisionPairs 构造.
    """

    env = GymUr5Env(port=19999, synchronous=True, max_steps=200)
    all_data = []
    skip_count = 0
    success_count = 0
    error_count = 0

    for ep in range(num_episodes):
        if verbose:
            print(f"\n=== Episode {ep + 1}/{num_episodes} ===")
        path = plan_path_for_scene(env, maxTime=maxTime, maxRetries=maxRetries, verbose=verbose)
        if len(path) < 2:
            skip_count += 1
            if verbose:
                print("[Warn] skip => path too short or planning fail!")
            # 如果是 ret=8( table contains non-numbers ), 这儿捕获不到,
            # 因为 ret=8会被 plan_path_for_scene(...) continue
            continue

        # 拆分关键帧 => (s,a)
        prev_j = np.array(path[0], dtype=np.float32)
        for i in range(1, len(path)):
            q_next = np.array(path[i], dtype=np.float32)
            s_i = build_state_from_joints(env, prev_j, real_tip_target)
            a_i = q_next - prev_j
            all_data.append((s_i, a_i))
            prev_j = q_next

        success_count += 1

    env.close()
    demos = np.array(all_data, dtype=object)
    np.savez(out_file, demos=demos)

    if verbose:
        print(f"[Collect] total_episodes={num_episodes}, skip={skip_count}, success={success_count}")
        print(f"[Collect] => {len(demos)} (s,a) pairs saved to {out_file}")


if __name__ == "__main__":
    # python collect_demos.py
    # 确保CoppeliaSim中载入UR5&脚本, 并已点▶运行
    collect_demonstrations(
        num_episodes=15000,
        out_file="demo_data.npz",
        real_tip_target=False,  # True时更真实, 但采集变慢
        maxTime=3.0,  # 与Lua中 local maxTime=3.0 一致
        maxRetries=3,
        verbose=True
    )
