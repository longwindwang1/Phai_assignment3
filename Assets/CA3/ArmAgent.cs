using Mujoco;
using UnityEngine;

public class ArmAgent : MonoBehaviour
{
    public int JointCount;
    public int StateSize;
    public int ActionSize;

    // ALOHA 2 names (from aloha.xml): left arm first, then right arm.
    public string[] jointNames =
    {
        "left/waist", "left/shoulder", "left/elbow", "left/forearm_roll",
        "left/wrist_angle", "left/wrist_rotate", "left/left_finger",
        "right/waist", "right/shoulder", "right/elbow", "right/forearm_roll",
        "right/wrist_angle", "right/wrist_rotate", "right/left_finger",
    };
    public string[] actuatorNames =
    {
        "left/waist", "left/shoulder", "left/elbow", "left/forearm_roll",
        "left/wrist_angle", "left/wrist_rotate", "left/gripper",
        "right/waist", "right/shoulder", "right/elbow", "right/forearm_roll",
        "right/wrist_angle", "right/wrist_rotate", "right/gripper",
    };

    public MjActuator gripper;
    public MjBody endEffectorBody;
    public Transform endEffector;
    public MjBody rightEndEffectorBody;
    public Transform rightEndEffector;
    // MjBaseJoint so the slide-joint grippers fit alongside the hinge joints.
    public MjBaseJoint[] joints;
    public MjActuator[] actuators;

    // Initial pose applied right after MuJoCo builds the scene (the plugin ignores MJCF keyframes
    // and starts at qpos = 0, where ALOHA's two grippers interpenetrate in the middle).
    // Default = aloha.xml "neutral_pose" keyframe, left arm then right arm.
    public bool startAtHome = true;
    public float[] startPose =
    {
        0f, -0.96f, 1.16f, 0f, -0.3f, 0f, 0.0084f,
        0f, -0.96f, 1.16f, 0f, -0.3f, 0f, 0.0084f,
    };
    // Mirror finger joints (coupled to left_finger by an equality constraint, not in `joints`).
    public string[] mirrorFingerNames = { "left/right_finger", "right/right_finger" };
    public int[] mirrorFingerSource = { 6, 13 };

    float[] action;
    float[] state;
    bool ready;

    public float[] CurrentAction => action;
    public bool Ready => ready;

    void Awake()
    {
        InitializeJointsAndActuators();
        if (ready && startAtHome)
            MjScene.Instance.postInitEvent += OnSceneCreated;
    }

    void OnDestroy()
    {
        if (MjScene.InstanceExists)
            MjScene.Instance.postInitEvent -= OnSceneCreated;
    }

    // Fired by MjScene after the model is compiled and before the first physics step.
    unsafe void OnSceneCreated(object sender, MjStepArgs args)
    {
        if (startPose == null || startPose.Length != JointCount)
        {
            Debug.LogWarning("ArmAgent: startPose must have " + JointCount + " entries; keeping qpos = 0.");
            return;
        }

        var data = args.data;
        for (int i = 0; i < JointCount; i++)
        {
            data->qpos[joints[i].QposAddress] = startPose[i];
            data->qvel[joints[i].DofAddress] = 0;
        }
        for (int k = 0; k < mirrorFingerNames.Length; k++)
        {
            var finger = FindComponent<MjBaseJoint>(mirrorFingerNames[k]);
            if (finger == null)
                continue;
            data->qpos[finger.QposAddress] = startPose[mirrorFingerSource[k]];
            data->qvel[finger.DofAddress] = 0;
        }
        // Recompute positions, contacts, etc. for the new state.
        MujocoLib.mj_forward(args.model, data);

        // Hold the start pose until a controller sets something else, so the first
        // ApplyAction does not pull the arms back towards 0.
        System.Array.Copy(startPose, action, ActionSize);
        ApplyAction();
        // MjScene copies actuator.Control into data->ctrl only *after* each mj_step, so the
        // very first step would still run with ctrl = 0 and yank the arms. Write ctrl directly.
        for (int i = 0; i < ActionSize; i++)
            data->ctrl[actuators[i].MujocoId] = action[i];
    }

    void FixedUpdate()
    {
        if (!ready)
            return;
        ApplyAction();
    }

    public void InitializeJointsAndActuators()
    {
        JointCount = jointNames.Length;
        ActionSize = actuatorNames.Length;
        StateSize = JointCount * 2;
        joints = new MjBaseJoint[JointCount];
        actuators = new MjActuator[ActionSize];
        action = new float[ActionSize];
        state = new float[StateSize];

        for (int i = 0; i < JointCount; i++)
            joints[i] = FindComponent<MjBaseJoint>(jointNames[i]);
        for (int i = 0; i < ActionSize; i++)
            actuators[i] = FindComponent<MjActuator>(actuatorNames[i]);

        endEffectorBody = FindComponent<MjBody>("left/gripper_link");
        rightEndEffectorBody = FindComponent<MjBody>("right/gripper_link");
        if (endEffectorBody != null)
            endEffector = endEffectorBody.transform;
        if (rightEndEffectorBody != null)
            rightEndEffector = rightEndEffectorBody.transform;
        if (gripper == null)
            gripper = FindComponent<MjActuator>("left/gripper");

        ready = System.Array.TrueForAll(joints, joint => joint != null) &&
                System.Array.TrueForAll(actuators, actuator => actuator != null) &&
                endEffectorBody != null &&
                rightEndEffectorBody != null;
        if (!ready)
            Debug.LogError("ArmAgent: missing joints/actuators/end effectors. Check the names.");
    }

    public void SetAction(float[] targets)
    {
        if (!ready || targets == null || targets.Length != ActionSize)
            return;
        System.Array.Copy(targets, action, ActionSize);
    }

    public void ApplyAction()
    {
        // Position actuators: Control is the target joint position (rad, or m for grippers).
        for (int i = 0; i < ActionSize; i++)
            actuators[i].Control = action[i];
    }

    public float[] Observe()
    {
        CopyState(state);
        return state;
    }

    public unsafe void CopyState(float[] destination)
    {
        if (!ready || destination == null || destination.Length != StateSize)
            return;
        if (!MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;
        var data = MjScene.Instance.Data;
        for (int i = 0; i < JointCount; i++)
        {
            destination[i] = (float)data->qpos[joints[i].QposAddress];
            destination[JointCount + i] = (float)data->qvel[joints[i].DofAddress];
        }
    }

    private T FindComponent<T>(string name) where T : Component
    {
        foreach (var c in FindObjectsByType<T>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            if (c.name == name)
                return c;
        Debug.LogWarning($"ArmAgent: {typeof(T).Name} '{name}' not found.");
        return null;
    }
}
