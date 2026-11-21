## Reflection

### 1

The hardest part was training the controller. The first time I used CMA-ES to search for the full set of parameters, none of the candidates outperformed the baseline after 60 rounds. After switching to joint-normalized parameters, the reward improved by 7.5% as expected, but once I integrated it into Unity, both arms kept oscillating near the target. I printed out the gains from `controller.json` and found that `k_pos` ranged from -4 to -9; the controller was “overshooting significantly,” exploiting a loophole in the world model that hadn’t encountered this type of motion before. Finally, following AI’s suggestion, I added `DEVIATION_PENALTY` to the reward in `rollout()` to penalize deviations from the target, which allowed me to strike a balance between stability and performance. The simplest part is the random target generation in the `ArmJointController`—all you need to do is sample and clip the values to the joint’s range.

### 2

I used RolloutRecorder in Unity to record 70,000 steps, with one line every 0.02 seconds. The model is an MLP with a structure of 42 → 256 → 256 → 28, and the activation function is SiLU. The input is the normalized [state, action] pair. I originally intended to predict the next state, but the AI suggested I output the normalized state change Δs instead of directly predicting the next state, because state changes are minimal within 0.02 seconds, and predicting the difference is easier to learn. During training, in addition to single-step MSE, I also added `multi_step_loss`, which allows the model to roll forward 10 steps using its own predictions before calculating the error, thereby reducing error accumulation during open-loop prediction. After the model was trained, I exported it to ONNX and used it to drive a semi-transparent ghost arm in WorldModelDriver.

### 3

The controller is a linear policy; when all parameters are set to 0, it is equivalent to “sending the target directly,” and CMA-ES only needs to learn the correction terms based on this. Training takes place entirely within the world model: `rollout()` samples some starting points from real-world data, changes to a random target every 150 steps, and has the world model simulate 200 steps in place of a physics simulation, then scores the result based on tracking error, velocity, motion jitter, and deviation from the target. In Unity, the `WorldModelController` calculates the position targets for all 14 joints using exactly the same formula at every physics step and passes them to MuJoCo’s actuator. To ensure consistency between training and deployment, I specifically changed Unity’s actionSmoothing from the default 0.10 to 0.2, matching the Python setting.

### 4

IkGoalProvider converts the positions of the two target balls in Cartesian space into target angles for each of the six rotational joints on each arm. I used Damped Least Squares (DLS), which requires solving only a 3×3 equation. I chose DLS because it is stable: when the target is outside the workspace or the robot arm is nearly fully extended, the damping term prevents the joint angles from becoming unbounded. There is an important detail in the implementation: initially, I was tracking the origin of `gripper_link`, but this caused the ball to be “swallowed” by the wrist; it only worked correctly after I switched to tracking the `site left/gripper` position between the fingertips.

### 5

I used to think that the more accurate the world model, the better the policy trained with it would be. This time, I realized that a world model is only reliable within the scope covered by the training data: since all my data followed the principle that “actions are approximately equal to goals,” when the controller learned to issue aggressive, overshoot-inducing commands, the model provided erroneously optimistic predictions. The fact that I expected a 7.5% improvement but observed oscillations in the actual simulation made me truly understand the “model exploitation” issue in model-based reinforcement learning. Additionally, watching the phantom arm move alongside the real arm gave me an intuitive understanding of how open-loop errors accumulate.

### 6

Specifically, I will design the data collection distribution from the outset to ensure it covers the states and actions that the policy might encounter later on, thereby avoiding the distribution shift issue we encountered this time. Most importantly, I will incorporate a regularization term or an uncertainty penalty into the training objective, and I will always evaluate “imagined rewards” and “actual simulation performance” separately, rather than relying solely on the former.