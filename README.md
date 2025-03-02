<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">

</head>
<body>

<h1>Demo2PPO: A UR5 RL Pipeline with Demonstration Collection and Behavior Cloning</h1>

<p>
  This repository showcases a pipeline for training a UR5 robotic arm in <strong>CoppeliaSim</strong> using 
  <em>demonstration collection</em>, <em>behavior cloning (BC)</em>, and <em>PPO reinforcement learning</em>.
</p>

<h2>Project Overview</h2>
<p>
  The main idea is to <strong>collect demonstrations</strong> using OMPL-based path planning 
  (<code>collect_demos.py</code>), then <strong>train a behavior cloning model</strong> to initialize the policy 
  (<code>behavior_cloning.py</code>), and finally use <strong>PPO</strong> to further fine-tune the policy 
  (<code>train_ppo.py</code>). 
</p>

<pre>
Project structure:
.
├─ collect_demos.py           # Collect (s, a) demonstration data using OMPL in CoppeliaSim
├─ behavior_cloning.py        # Offline behavior cloning to learn a policy from demonstrations
├─ train_ppo.py               # Use Stable-Baselines3 to train a PPO agent on UR5
├─ ppo2/                      # Contains GymUr5Env and utility scripts
├─ sim.py / simConst.py       # CoppeliaSim Python Remote API (needed for script import)
└─ README.md                  # This document
</pre>

<h2>Dependencies</h2>
<ul>
  <li><strong>Python 3.7+</strong> (3.8+ recommended)</li>
  <li><strong>CoppeliaSim</strong> (already installed and running)</li>
  <li><strong>Legacy Remote API scripts</strong> (sim.py, simConst.py)</li>
  <li><strong>NumPy, PyTorch, Stable-Baselines3, Gymnasium</strong></li>
</ul>

<pre>pip install numpy torch stable-baselines3 gymnasium</pre>

<p>
  Ensure the <strong>Legacy Remote API</strong> server is active on port <code>19999</code> in CoppeliaSim, 
  and that you have pressed the 'Play' button (<strong>▶</strong>) in CoppeliaSim's GUI.
</p>

<h2>Usage</h2>

<h3>1. Demonstration Collection (<code>collect_demos.py</code>)</h3>
<p>
  This script invokes a <em>Lua</em> function <code>computePathOMPLLua</code> to generate OMPL paths. 
  It dissects each path into state-action pairs (<code>(s, a)</code>) and saves them in an <code>.npz</code> file.
</p>

<pre>python collect_demos.py</pre>

<ul>
  <li><strong>num_episodes:</strong> How many times OMPL tries to plan and generate a path</li>
  <li><strong>out_file:</strong> Output <code>.npz</code> file (e.g., <code>demo_data.npz</code>)</li>
  <li><strong>real_tip_target:</strong> Whether to update tip/target positions in real time</li>
</ul>

<p>
  After the process finishes, you will find an <code>.npz</code> file (e.g., <code>demo_data.npz</code>) 
  containing all collected <code>(s, a)</code> pairs.
</p>

<h3>2. Offline Behavior Cloning (<code>behavior_cloning.py</code>)</h3>
<p>
  This script trains a <strong>2-layer MLP</strong> (similar to <code>MlpPolicy</code> in SB3) on the collected 
  demonstration data. It saves the network weights in <code>bc_model.pth</code>.
</p>

<pre>python behavior_cloning.py --demo_file=demo_data.npz \
                            --bc_model_file=bc_model.pth \
                            --max_epochs=20 \
                            --batch_size=64 \
                            --lr=1e-3
</pre>

<ul>
  <li><strong>--demo_file:</strong> Path to the demonstration data (<code>.npz</code>)</li>
  <li><strong>--bc_model_file:</strong> Where to save the learned weights (<code>.pth</code>)</li>
  <li><strong>--max_epochs, --batch_size, --lr:</strong> Model hyperparameters</li>
</ul>

<h3>3. PPO Training (<code>train_ppo.py</code>)</h3>
<p>
  This script demonstrates how to <strong>train a PPO agent</strong> in the <code>GymUr5Env</code> environment 
  (built on top of <code>UR5Env</code>), using <em>Stable-Baselines3</em>. It frequently calls a custom 
  <em>EvalCallback</em> to measure success rate, collision rate, etc.
</p>

<pre>python train_ppo.py</pre>

<p>
  Key hyperparameters (e.g., <code>total_timesteps</code>, <code>learning_rate</code>, <code>batch_size</code>) 
  can be adjusted in the script.
</p>
<p>
  The script will log training metrics to a specified directory (e.g., 
  <code>D:\program\PPOrobot\tensorboard_logs</code>). 
  You can launch <em>TensorBoard</em> to visualize results:
</p>

<pre>tensorboard --logdir="D:\program\PPOrobot\tensorboard_logs"</pre>

<p>
  After training completes, a final model <code>ppo_ur5_model</code> is saved, and best checkpoints 
  (based on the evaluation callback) can be found in the <code>best_model</code> directory.
</p>

<h2>Validation and Deployment</h2>
<ul>
  <li><strong>In-script test:</strong> After <code>train_ppo.py</code> finishes, it performs a quick test run.</li>
  <li><strong>Loading the PPO policy:</strong> You can load your saved model via 
    <code>model = PPO.load("ppo_ur5_model")</code> and then <code>model.predict(obs)</code> in your own scripts.</li>
  <li><strong>Loading BC policy:</strong> Use PyTorch to load <code>bc_model.pth</code> and evaluate 
    <code>model.forward(state_tensor)</code> for inference.</li>
</ul>

<h2>Important Notes</h2>
<ul>
  <li>Ensure that <strong>CoppeliaSim is in Play mode</strong>. If it's not running, the scripts will fail or hang.</li>
  <li>If you encounter <code>ret=8 (table contains non-numbers)</code> during <code>simxCallScriptFunction</code>, 
      check your Lua script in CoppeliaSim for any OMPL configurations or data type issues.</li>
  <li>The environment uses <em>synchronous mode</em> if available (<code>sim.simxSynchronous</code>), 
      otherwise falls back to asynchronous mode.</li>
  <li>The reward function is relatively simple (distance-based plus collision penalty). 
      You may customize it depending on your tasks.</li>
</ul>

<h2>Q&amp;A</h2>
<dl>
  <dt><strong>Q1</strong>: Can I change the log and model save directories?</dt>
  <dd>
    <strong>A1</strong>: Yes. In <code>train_ppo.py</code>, edit <code>ROOT_DIR</code>, 
    <code>TENSORBOARD_DIR</code>, <code>EVAL_LOG_DIR</code>, <code>BEST_MODEL_DIR</code>, etc.
  </dd>

  <dt><strong>Q2</strong>: How do I initialize the PPO policy with my BC model?</dt>
  <dd>
    <strong>A2</strong>: One approach is to create a <code>PPO("MlpPolicy", ...)</code> object, 
    then manually load the BC network weights into <code>model.policy</code>. 
    Check SB3 documentation or community forums for examples.
  </dd>

  <dt><strong>Q3</strong>: What about joint angle limits in <code>UR5Env</code>?</dt>
  <dd>
    <strong>A3</strong>: Adjust the arrays 
    <code>self.joint_limits_low</code> and <code>self.joint_limits_high</code> 
    in the <code>UR5Env</code> constructor to fit your needs.
  </dd>
</dl>

<h2>References</h2>
<ul>
  <li><a href="https://www.coppeliarobotics.com/docs/">CoppeliaSim Documentation</a></li>
  <li><a href="https://stable-baselines3.readthedocs.io/">Stable-Baselines3 Documentation</a></li>
  <li><a href="https://pytorch.org/docs/stable/">PyTorch Documentation</a></li>
</ul>

<p><em>Enjoy your journey in RL with CoppeliaSim!</em></p>

</body>
</html>
