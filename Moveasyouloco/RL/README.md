
---

# Reinforcement Learning Configuration Parameters for Imitation Learning in MuJoCo

_Focus: Joint Torque Analysis in SkeletonTorque/SkeletonMuscle Environments_

---

## 1. Environment Parameters

_Parameters defining the simulation environment, physics, and agent morphology._

---

### **`env_name`**

- **Description**: Name of the MuJoCo environment. Determines the agent's morphology (e.g., humanoid, quadruped) and physics (e.g., muscle actuations vs. torque control).
- **Typical Range**: _`MjxSkeletonMuscle`, `MjxUnitreeH1`, `MjxSkeletonTorque`_
- **Effect**: Core to the task; `SkeletonMuscle` includes muscle dynamics, while `SkeletonTorque` uses direct torque control.

---

### **`use_mjwarp`**

- **Description**: Enables MuJoCo Warp, a GPU-accelerated version of MuJoCo for faster simulations.
- **Typical Range**: _`True`, `False`_
- **Effect**: Significantly speeds up training but requires NVIDIA GPU and compatible MuJoCo version.

---

### **`nconmax`**

- **Description**: Maximum number of contact points allowed in the simulation. Limits the complexity of collision detection.
- **Typical Range**: _Based on current number estimate: `5000–20000`_
- **Effect**: Higher values allow more complex interactions (e.g., multi-limb contact) but increase computational cost. Too low may cause simulation instabilities.

---

### **`njmax`**

- **Description**: Maximum number of joints in the model. Limits the degrees of freedom (DoF) of the agent.
- **Typical Range**: _Based on current number estimate: `50–200`_
- **Effect**: Must match the environment’s actual joint count. Too low truncates the model; too high wastes memory.


---

### **`headless`**

- **Description**: Runs the simulation without rendering (no GUI).
- **Typical Range**: _`True`, `False`_
- **Effect**: Improves speed for training but disables visualization. Use `False` for debugging.

---

### **`horizon`**

- **Description**: Maximum length (in timesteps) of an episode.
- **Typical Range**: _`500–5000`_
- **Effect**: Longer horizons allow for more complex behaviors but increase memory usage and may require tuning `gamma` and `gae_lambda`.

---

### **`init_state_type`**

- **Description**: Method for initializing the agent’s state at the start of each episode (e.g., `TrajInitialStateHandler` for imitation learning).
- **Typical Range**: _`TrajInitialStateHandler`, `DefaultInitialStateHandler`_
- **Effect**: Critical for imitation learning; `TrajInitialStateHandler` samples initial states from reference motion data.

---

## 2. Goal and Reward Parameters

_Parameters defining the task objectives, reward functions, and mimicry targets._

---

### **`goal_type`**

- **Description**: Type of goal used for the task (e.g., trajectory mimicry).
- **Typical Range**: _`GoalTrajMimic`, `GoalPosMimic`_
- **Effect**: `GoalTrajMimic` enforces full trajectory imitation, while `GoalPosMimic` focuses on positional accuracy.

---

### **`goal_params`**

#### **`visualize_goal`**

- **Description**: Whether to render the goal (e.g., target trajectory) in the simulation.
- **Typical Range**: _`True`, `False`_
- **Effect**: Useful for debugging but may slow down training.

---

### **`reward_type`**

- **Description**: Type of reward function used to evaluate the agent’s performance.
- **Typical Range**: _`MimicReward`_
- **Effect**: `MimicReward` computes rewards based on the similarity between the agent’s motion and reference data.

---

### **`reward_params`**

#### **`qpos_w_sum`**

- **Description**: Weight for the sum of joint position errors in the reward function.
- **Typical Range**: _`0.1–1.0`_
- **Effect**: Higher values penalize deviations in joint positions more strongly. Critical for imitation accuracy.

---

#### **`qvel_w_sum`**

- **Description**: Weight for the sum of joint velocity errors.
- **Typical Range**: _`0.1–0.5`_
- **Effect**: Balances positional and velocity accuracy; too high may cause jittery motions.

---

#### **`rpos_w_sum`**

- **Description**: Weight for the sum of root (pelvis) position errors.
- **Typical Range**: _`0.3–1.0`_
- **Effect**: Ensures the agent’s root (e.g., pelvis) follows the reference trajectory closely.

---

#### **`rquat_w_sum`**

- **Description**: Weight for the sum of root orientation (quaternion) errors.
- **Typical Range**: _`0.1–0.5`_
- **Effect**: Penalizes misalignment in the agent’s global orientation.

---

#### **`rvel_w_sum`**

- **Description**: Weight for the sum of root velocity errors.
- **Typical Range**: _`0.01–0.2`_
- **Effect**: Smooths root motion; too high may conflict with `rpos_w_sum`.

---

#### **`qpos_w_exp`**

- **Description**: Weight for the exponential of joint position errors (emphasizes large errors).
- **Typical Range**: _`0.0–0.5`_
- **Effect**: Non-linear penalty; useful for avoiding extreme deviations.

---

#### **`qvel_w_exp`**

- **Description**: Weight for the exponential of joint velocity errors.
- **Typical Range**: _`0.0–0.3`_
- **Effect**: Similar to `qpos_w_exp` but for velocities.

---

#### **`rpos_w_exp`**

- **Description**: Weight for the exponential of root position errors.
- **Typical Range**: _`0.0–0.5`_
- **Effect**: Strongly penalizes large positional deviations of the root.

---

#### **`rquat_w_exp`**

- **Description**: Weight for the exponential of root orientation errors.
- **Typical Range**: _`0.0–0.3`_
- **Effect**: Emphasizes large orientation mistakes.

---

#### **`rvel_w_exp`**

- **Description**: Weight for the exponential of root velocity errors.
- **Typical Range**: _`0.0–0.1`_
- **Effect**: Penalizes large root velocity mismatches.

---

#### **`action_out_of_bounds_coeff`**

- **Description**: Penalty coefficient for actions outside the valid range (e.g., torque limits).
- **Typical Range**: _`0.0–10.0`_
- **Effect**: Discourages invalid actions; set to `0` to disable.

---

#### **`joint_acc_coeff`**

- **Description**: Penalty coefficient for high joint accelerations (smoothness regularization).
- **Typical Range**: _`0.0–0.1`_
- **Effect**: Reduces jerky motions by penalizing rapid acceleration changes.

---

#### **`joint_torque_coeff`**

- **Description**: Penalty coefficient for high joint torques (effort regularization).
- **Typical Range**: _`0.0–0.01`_
- **Effect**: Encourages energy-efficient motions by penalizing excessive torque.

---

#### **`action_rate_coeff`**

- **Description**: Penalty coefficient for rapid changes in actions (temporal smoothness).
- **Typical Range**: _`0.0–0.1`_
- **Effect**: Smooths the policy’s output over time.

---

#### **`sites_for_mimic`**

- **Description**: List of body sites (markers) used for mimicry rewards. These are typically 3D points on the agent’s body (e.g., hands, feet).
- **Typical Range**:
    - _`upper_body_mimic`_
    - _`left_hand_mimic`_
    - _`left_foot_mimic`_
    - _`right_hand_mimic`_
    - _`right_foot_mimic`_
    - _`pelvis_mimic`_
    - _`head_mimic`_
    - _`right_shoulder_mimic`_
    - _`right_elbow_mimic`_
    - _`left_shoulder_mimic`_
    - _`left_elbow_mimic`_
    - _`right_hip_mimic`_
    - _`right_knee_mimic`_
    - _`left_hip_mimic`_
    - _`left_knee_mimic`_
- **Effect**: Defines which body parts are prioritized for imitation. More sites increase reward computation cost.

---

#### **`joints_for_mimic`**

- **Description**: List of joints used for mimicry rewards (alternative to `sites_for_mimic`).
- **Typical Range**: _List of joint names (e.g., `hip_flexion_r`, `knee_angle_l`)_
- **Effect**: Focuses imitation on specific joints rather than body sites.

---

#### **`joints_to_ignore`**

- **Description**: List of joints excluded from reward calculations or control.
- **Typical Range**:
    - _`pelvis_tx`_
    - _`pelvis_tz`_
    - _`pelvis_ty`_
    - _`pelvis_tilt`_
    - _`pelvis_list`_
    - _`pelvis_rotation`_
- **Effect**: Ignores specified joints (e.g., pelvis DoFs) to reduce complexity or avoid overfitting to irrelevant motions.

---

## 3. Training Hyperparameters

_Parameters controlling the learning process, optimization, and stability._

---

### **`hidden_layers`**

- **Description**: List of integers defining the size of hidden layers in the neural network (policy/critic).
- **Typical Range**: _`[64, 64]`, `[256, 128]`, `[512, 256]`_
- **Effect**: Deeper/larger networks increase capacity but may require more data and tuning. **Warning**: Very large layers (e.g., `[1024, 512]`) may cause memory issues.

---

### **`lr`**

- **Description**: Learning rate for the optimizer (e.g., Adam).
- **Typical Range**: _`1e-5–1e-3`_
- **Effect**: Higher values speed up learning but may cause instability. **Warning**: Start with `1e-4` and adjust based on reward curves.

---

### **`num_envs`**

- **Description**: Number of parallel environments for training.
- **Typical Range**: _`16–8192`_
- **Effect**: More environments improve sample efficiency but increase memory/CPU usage.

---

### **`num_steps`**

- **Description**: Number of steps (actions) taken in each environment per rollout.
- **Typical Range**: _`100–1000`_
- **Effect**: Longer rollouts provide more diverse data but may reduce update frequency.

---

### **`record_n_steps`**

- **Description**: Number of steps to record for logging/visualization (e.g., for videos).
- **Typical Range**: _`100–1000`_
- **Effect**: Only affects logging; does not impact training.

---

### **`total_timesteps`**

- **Description**: Total number of timesteps for training (across all environments).
- **Typical Range**: _`1e6–1e9`_
- **Effect**: Longer training may improve performance but increases computational cost. **Warning**: For `SkeletonMuscle`, start with `1e7–1e8` due to higher complexity.

---

### **`update_epochs`**

- **Description**: Number of epochs (passes over the rollout data) per policy update.
- **Typical Range**: _`4–16`_
- **Effect**: More epochs can improve sample efficiency but may lead to overfitting. **Warning**: High values (e.g., `>10`) may require reducing `lr`.

---

### **`proportion_env_reward`**

- **Description**: Fraction of the reward used for environment-specific objectives (e.g., survival bonuses).
- **Typical Range**: _`0.0–0.5`_
- **Effect**: `0.0` means only mimicry rewards are used.

---

### **`num_minibatches`**

- **Description**: Number of minibatches for stochastic gradient descent (SGD) updates.
- **Typical Range**: _`32–512`_
- **Effect**: More minibatches stabilize training but increase computation per update.

---

### **`gamma`**

- **Description**: Discount factor for future rewards in the return calculation.
- **Typical Range**: _`0.95–0.999`_
- **Effect**: Higher values prioritize long-term rewards. **Warning**: For long horizons (e.g., `>2000`), use `gamma` close to `0.99`.

---

### **`gae_lambda`**

- **Description**: Lambda parameter for Generalized Advantage Estimation (GAE). Controls the bias-variance tradeoff of advantage estimates.
- **Typical Range**: _`0.8–0.99`_
- **Effect**: Higher values reduce bias but increase variance. Typically set close to `gamma`.

---

### **`clip_eps`**

- **Description**: Clipping range for Proximal Policy Optimization (PPO) updates.
- **Typical Range**: _`0.1–0.3`_
- **Effect**: Limits policy updates to improve stability. **Warning**: Too high (e.g., `>0.5`) may cause instability.

---

### **`init_std`**

- **Description**: Initial standard deviation for the policy’s action distribution (e.g., Gaussian).
- **Typical Range**: _`0.1–0.5`_
- **Effect**: Higher values encourage initial exploration but may slow early learning.

---

### **`learnable_std`**

- **Description**: Whether the standard deviation of the action distribution is learnable.
- **Typical Range**: _`True`, `False`_
- **Effect**: `True` allows the agent to adapt exploration dynamically but adds complexity.

---

### **`ent_coef`**

- **Description**: Coefficient for the entropy bonus in the policy loss. Encourages exploration.
- **Typical Range**: _`0.0–0.1`_
- **Effect**: Higher values increase exploration but may reduce focus on reward optimization.

---

### **`vf_coef`**

- **Description**: Coefficient for the value function loss in PPO.
- **Typical Range**: _`0.1–1.0`_
- **Effect**: Balances policy and value function updates. Typically set to `0.5`.

---

### **`max_grad_norm`**

- **Description**: Maximum norm for gradient clipping.
- **Typical Range**: _`0.1–1.0`_
- **Effect**: Prevents exploding gradients. **Warning**: Too high (e.g., `>1.0`) may reduce the effect of clipping.

---

### **`activation`**

- **Description**: Activation function for the neural network layers.
- **Typical Range**: _`tanh`, `relu`, `leaky_relu`_
- **Effect**: `tanh` is common for policy networks (bounded outputs), while `relu` may speed up training.

---

### **`anneal_lr`**

- **Description**: Whether to anneal (linearly decay) the learning rate over training.
- **Typical Range**: _`True`, `False`_
- **Effect**: Can improve convergence but may slow initial learning.

---

### **`weight_decay`**

- **Description**: L2 regularization coefficient for the network weights.
- **Typical Range**: _`0.0–0.01`_
- **Effect**: Reduces overfitting but may slow learning if too high.

---

### **`normalize_env`**

- **Description**: Whether to normalize observations and rewards.
- **Typical Range**: _`True`, `False`_
- **Effect**: Improves stability but may require careful tuning for imitation tasks.

---

### **`debug`**

- **Description**: Enables debug mode (e.g., additional logging, checks).
- **Typical Range**: _`True`, `False`_
- **Effect**: Useful for development but slows down training.

---

### **`n_seeds`**

- **Description**: Number of random seeds to run for reproducibility.
- **Typical Range**: _`1–10`_
- **Effect**: More seeds improve statistical significance but increase computational cost.

---

### **`vmap_across_seeds`**

- **Description**: Whether to vectorize operations across seeds for efficiency.
- **Typical Range**: _`True`, `False`_
- **Effect**: Speeds up multi-seed training but requires compatible hardware.

---

## 4. Control Parameters

_Parameters defining the control scheme (e.g., PD control, torque control) for the agent._

---

### **`control_type`**

- **Description**: Type of control used to map actions to joint torques/positions.
- **Typical Range**:
    - _`PDControl`_ (Proportional-Derivative control)
    - _`DefaultControl`_ (Direct torque control)
- **Effect**: `PDControl` is common for position control, while `DefaultControl` is used for torque-based environments like `SkeletonTorque`.

---

### **`control_params`**

#### **For `PDControl`:**

##### **`p_gain`**

- **Description**: Proportional gain for each joint’s PD controller. Scales the position error term.
- **Typical Range**: _`50–500` (joint-dependent)_
- **Effect**: Higher values increase stiffness (faster response to errors but may cause oscillations). **Warning**: Start with lower values (e.g., `100`) and tune per joint.

##### **`d_gain`**

- **Description**: Derivative gain for each joint’s PD controller. Scales the velocity error term.
- **Typical Range**: _`1–10` (joint-dependent)_
- **Effect**: Dampens oscillations; too high may slow response. Typically `d_gain = p_gain / 10`.

---

#### **For `DefaultControl`:**

- **Description**: No additional parameters; actions are directly mapped to torques.

---

## 5. Dataset and Task Parameters

_Parameters related to reference data, task setup, and dataset configuration._

---

### **`task_factory`**

#### **`name`**

- **Description**: Name of the task factory (e.g., `ImitationFactory` for imitation learning).
- **Typical Range**: _`ImitationFactory`_
- **Effect**: Defines the task type (e.g., imitation vs. reinforcement learning).

---

#### **`params`**

##### **`custom_dataset_conf`**

- **Description**: Configuration for custom datasets (e.g., path to motion data).
- **Typical Range**: _Path to `.npz` file (e.g., `"Moveasyouloco/Data_Conversion/Output_Files/squat5_converted.npz"`)_
- **Effect**: Specifies the reference motion data for imitation.

---

##### **`traj`**

- **Description**: Path to the trajectory file for imitation.
- **Typical Range**: _Path to `.npz` file_
- **Effect**: Overrides `custom_dataset_conf` if provided.

---

##### **`lafan1_dataset_conf`**

- **Description**: Configuration for LaFAN1 dataset (humanoid motions).
- **Typical Range**: _`dataset_name: ["walk1_subject5", ...]`_
- **Effect**: Used for pre-loaded motion datasets.

---

##### **`default_dataset_conf`**

- **Description**: Default dataset configuration (e.g., for generic motions).
- **Typical Range**: _Dictionary of dataset parameters_
- **Effect**: Fallback if no custom dataset is specified.

---

#### **`amass_dataset_conf`**

- **Description**: Configuration for AMASS dataset (large-scale motion capture data).
- **Typical Range**: _Dictionary of dataset parameters_
- **Effect**: Used for high-diversity motion imitation.

---

## 6. Validation Parameters

_Parameters controlling validation, logging, and evaluation during training._

---

### **`validation`**

#### **`active`**

- **Description**: Whether to run validation during training.
- **Typical Range**: _`True`, `False`_
- **Effect**: Adds computational overhead but helps monitor generalization.

---

#### **`num_steps`**

- **Description**: Number of steps per validation episode.
- **Typical Range**: _`100–1000`_
- **Effect**: Should match `num_steps` in training for consistency.

---

#### **`num_envs`**

- **Description**: Number of parallel environments for validation.
- **Typical Range**: _`1–100`_
- **Effect**: More environments provide more robust validation but increase cost.

---

#### **`num`**

- **Description**: Number of validation episodes to run.
- **Typical Range**: _`1–50`_
- **Effect**: Set to `0` to disable validation.

---

#### **`quantities`**

- **Description**: List of quantities to compute for validation (e.g., joint positions, velocities).
- **Typical Range**:
    - _`JointPosition`_
    - _`JointVelocity`_
    - _`RelSitePosition`_
    - _`RelSiteVelocity`_
    - _`RelSiteOrientation`_
- **Effect**: Determines which metrics are logged for analysis.

---

#### **`measures`**

- **Description**: List of distance measures for validation (e.g., to compare against reference motion).
- **Typical Range**:
    - _`EuclideanDistance`_
    - _`DynamicTimeWarping`_
    - _`DiscreteFrechetDistance`_
- **Effect**: `DynamicTimeWarping` is robust to temporal misalignments; `EuclideanDistance` is simpler but sensitive to phase shifts.

---

#### **`rel_site_names`**

- **Description**: List of body sites for validation metrics (same as `sites_for_mimic`).
- **Typical Range**: _See `sites_for_mimic`_
- **Effect**: Must match the sites used in training for consistent evaluation.

---

#### **`validation_interval`**

- **Description**: Number of training updates between validation runs.
- **Typical Range**: _Based on current number estimate: `10–1000`_
- **Effect**: More frequent validation provides better monitoring but slows training.

---

#### **`num_updates`**

- **Description**: Total number of policy updates during training.
- **Typical Range**: _Based on current number estimate: `1000–1e6`_
- **Effect**: Computed from `total_timesteps`, `num_steps`, and `num_envs`.

---

#### **`minibatch_size`**

- **Description**: Size of each minibatch for SGD updates.
- **Typical Range**: _Based on current number estimate: `32–1024`_
- **Effect**: Computed from `num_envs`, `num_steps`, and `num_minibatches`.

---

## 7. Logging and Experiment Tracking

_Parameters for logging, visualization, and experiment management._

---

### **`wandb`**

#### **`project`**

- **Description**: Name of the Weights & Biases (wandb) project for logging.
- **Typical Range**: _`"deepmimic"`, `"imitation_learning"`_
- **Effect**: Organizes experiments in wandb dashboard.

---

## 8. Observation and History Parameters

_Parameters related to observations, history, and input to the policy._

---

### **`actor_obs_group`**

- **Description**: Name of the observation group used by the actor (policy).
- **Typical Range**: _`"policy"`, `"default"`_
- **Effect**: Defines which observations are fed to the policy network.

---

### **`critic_obs_group`**

- **Description**: Name of the observation group used by the critic (value function).
- **Typical Range**: _`"policy"`, `"default"`_
- **Effect**: May differ from `actor_obs_group` if the critic requires additional information.

---

### **`len_obs_history`**

- **Description**: Length of the observation history stack provided to the policy.
- **Typical Range**: _`1–10`_
- **Effect**: Longer histories provide temporal context but increase input dimensionality.

---

## 9. Hydra Configuration (Optional)

_Parameters for Hydra (experiment management framework)._

---

### **`hydra`**

#### **`mode`**

- **Description**: Hydra mode (e.g., `MULTIRUN` for sweeping parameters).
- **Typical Range**: _`MULTIRUN`, `RUN`_
- **Effect**: `MULTIRUN` enables parameter sweeps.

---

#### **`job_logging`**

- **Description**: Configuration for Hydra’s job logging.
- **Typical Range**: _`default`, custom dict_
- **Effect**: Controls where and how logs are saved.

---

#### **`hydra_logging`**

- **Description**: Configuration for Hydra’s internal logging.
- **Typical Range**: _`default`, custom dict_
- **Effect**: Separate from job logging; controls Hydra’s own logs.

---

#### **`sweeper`**

- **Description**: Configuration for Hydra’s parameter sweeper.
- **Typical Range**: _Custom dict (e.g., `params: experiment.lr: 1e-4, 1e-3`)_
- **Effect**: Defines parameter grids for `MULTIRUN`.

---

#### **`sweep`**

##### **`params`**

- **Description**: Parameters to sweep in `MULTIRUN` mode.
- **Typical Range**: _Dictionary of parameter lists (e.g., `experiment.learnable_std: true, false`)_
- **Effect**: Enables hyperparameter tuning.

---

## 10. Miscellaneous Parameters

_Less common or advanced parameters._

---

### **`custom_dataset_conf`**

- **Description**: Custom dataset configuration (e.g., for non-standard motion data).
- **Typical Range**: _Dictionary of dataset parameters_
- **Effect**: Overrides default dataset settings.

---

## 11. Final Notes

- **Critical Parameters for Imitation Learning**:
    - Reward weights (`qpos_w_sum`, `rpos_w_sum`, etc.) must be tuned carefully to balance motion accuracy.
    - `horizon` and `num_steps` should match the length of reference motions.
    - For `SkeletonMuscle`, `nconmax` and `njmax` may need to be higher due to muscle complexity.
- **Stability Warnings**:
    - High `lr` + high `update_epochs` can cause instability.
    - Low `nconmax`/`njmax` may crash simulations with complex contacts.
    - `learnable_std = True` may require lower `init_std` to avoid initial chaos.