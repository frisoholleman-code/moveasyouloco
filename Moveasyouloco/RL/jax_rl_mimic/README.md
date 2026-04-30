## DeepMimic Imitation Learning Example

This example demonstrates training a PPO agent on the Unitree H1 robot using a DeepMimic‑style reward
to imitate a dataset of human walking trajectories in various directions.

The example leverages the `GoalTrajMimic` goal vector, which encodes target positions, orientations, and velocities
of both site and joint states from the expert dataset. The reward function defined in the configuration, `MimicReward`,
implements DeepMimic‑style reward. By the end of training, the agent can walk in multiple directions and rotate 
around its z‑axis.

---

### 🚀 Training

To train the agent, run:

```bash
python experiment.py
```

This command will:

- Train the PPO agent on the Unitree H1 robot for 300 million steps (approximately 36 minutes on an RTX 3080 Ti). To get even smoother policies train it for longer!
- Save the trained agent (as `GAILJax_saved.pkl` in the `outputs` folder).
- Perform a final rendering of the trained policy.
- Save a video of the rendering to the `LocoMuJoCo_recordings/` directory.
- Upload the video to Weights & Biases (WandB) for further analysis (check the command line logs for details).


#### Validation Loop During Training

Throughout training, the agent will be evaluated using various trajectory-based metrics, including 
Euclidean distance, Dynamic Time Warping (DTW), and discrete Fréchet distance. These metrics will be 
computed on different entities such as joint positions, joint velocities, and site positions and orientations. 
All results will be logged to Weights & Biases (WandB).

---

### 📈 Evaluation

To evaluate the trained agent, run:

```bash
python eval.py --path path/to/agent_file
```

If you'd like to evaluate the agent using MuJoCo (instead of Mjx), run:

```bash
python eval.py --path path/to/agent_file --use_mujoco
```

#### Joint Torque Analysis

To evaluate and analyze the joint torques produced by the trained agent, run:

```bash
python eval_torques.py --path path/to/agent_file --n_steps 1000 --save_torques
```

This will:
- Run the agent for the specified number of steps using MuJoCo
- Collect joint torque data from all actuators
- Display detailed statistics (mean, max, RMS) for each joint
- Optionally save the torque data to a `.npz` file for further analysis

Parameters:
- `--path`: Path to the trained agent file (required)
- `--n_steps`: Number of evaluation steps (default: 1000)
- `--save_torques`: Save torque data to file (optional)

> ⚠️ **Note:** Evaluating with MuJoCo may not yield results as robust as with Mjx due to simulator differences. For reliable policy transfer between the two, consider applying domain randomization techniques.
