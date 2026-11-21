using System;
using UnityEngine;


[DefaultExecutionOrder(-5)]
public class WorldModelController : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController goals;
    public TextAsset controllerJson;

    // Must match ACTION_SMOOTH / MAX_DELTA in train_controller.py.
    [Range(0.0f, 1f)]
    public float actionSmoothing = 0.2f;

    public float maxDeltaPerStep = 0.05f;

    float[] W;
    float[] b;
    float[] jointLo;
    float[] jointHi;

    float[] features;
    float[] rawAction;
    float[] action;
    float[] state;
    float[] goalBuffer;
    bool ready;
    bool hasAction;
    bool tookOverEmitActions;
    bool previousEmitActions;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (goals == null)
            goals = GetComponent<ArmJointController>();
        if (agent == null || goals == null || controllerJson == null)
        {
            Debug.LogError("WorldModelController needs agent, goals, and controller.json.");
            enabled = false;
            return;
        }

        features = new float[agent.StateSize];
        rawAction = new float[agent.ActionSize];
        action = new float[agent.ActionSize];
        state = new float[agent.StateSize];
        goalBuffer = new float[agent.ActionSize];

        if (!Load(controllerJson.text))
        {
            enabled = false;
            return;
        }

        previousEmitActions = goals.emitActions;
        goals.emitActions = false;
        tookOverEmitActions = true;
        ready = true;
        Debug.Log(
            "WorldModelController ready (CMA-ES Controller C). " +
            "Goals come from ArmJointController; actions come from the learned linear policy.");
    }

    void OnEnable()
    {
        if (!ready || goals == null)
            return;

        previousEmitActions = goals.emitActions;
        goals.emitActions = false;
        tookOverEmitActions = true;
        hasAction = false;
    }

    void OnDisable()
    {
        if (tookOverEmitActions && goals != null)
        {
            goals.emitActions = previousEmitActions;
            tookOverEmitActions = false;
        }
        hasAction = false;
    }

    // Thanks for the instruction by claude
    void FixedUpdate()
    {
        if (!ready || !agent.Ready)
            return;

        agent.CopyState(state);

        // First step: start smoothing from the current joint angles (Python: prev = state[:ACTION]),
        // otherwise the arm would be pulled towards 0 on the first frame.
        if (!hasAction)
        {
            Array.Copy(state, action, agent.ActionSize);
            hasAction = true;
        }

        ActionFrom(state, rawAction);        // goal comes from ArmJointController (random or IK)
        ApplySmoothing(rawAction, action);   // EMA + per-step clamp + joint limits
        agent.SetAction(action);
    }

    // Thanks for the instruction by claude
    public void ActionFrom(float[] state, float[] destination)
    {
        // Current goal from ArmJointController, then the policy. Also used by WorldModelDriver
        // to run the policy on the imagined (ghost) state.
        if (!ready || goals == null || goalBuffer == null)
            return;
        goals.ActionFrom(goalBuffer);
        ActionFrom(state, goalBuffer, destination);
    }

    // Thanks for the instruction by claude
    public void ActionFrom(float[] state, float[] goal, float[] destination)
    {
        // Mirror of train_controller.policy():
        //   f = [q - goal, qd],  action = clip(goal + W f + b, joint_lo, joint_hi)
        // W is stored row-major (ActionSize x StateSize): W[i, j] = W[i * StateSize + j].
        if (!ready)
            return;
        int a = agent.ActionSize;
        int s = agent.StateSize;

        for (int i = 0; i < a; i++)
        {
            features[i] = state[i] - goal[i];
            features[a + i] = state[a + i];
        }

        for (int i = 0; i < a; i++)
        {
            float sum = b[i];
            int row = i * s;
            for (int j = 0; j < s; j++)
                sum += W[row + j] * features[j];
            destination[i] = Mathf.Clamp(goal[i] + sum, jointLo[i], jointHi[i]);
        }
    }

    // Thanks for the instruction by claude
    void ApplySmoothing(float[] raw, float[] smoothed)
    {
        // Mirror of train_controller.smooth_action(); `smoothed` holds the previous action
        // and is updated in place:
        //   blended = (1 - s) * prev + s * raw
        //   delta   = clip(blended - prev, -maxDelta, maxDelta)
        //   out     = clip(prev + delta, joint_lo, joint_hi)
        for (int i = 0; i < agent.ActionSize; i++)
        {
            float prev = smoothed[i];
            float blended = (1f - actionSmoothing) * prev + actionSmoothing * raw[i];
            float delta = Mathf.Clamp(blended - prev, -maxDeltaPerStep, maxDeltaPerStep);
            smoothed[i] = Mathf.Clamp(prev + delta, jointLo[i], jointHi[i]);
        }
    }

    bool Load(string json)
    {
        try
        {
            var data = JsonUtility.FromJson<ControllerFile>(json);
            if (data == null || data.W == null || data.b == null ||
                data.W.Length != agent.ActionSize * agent.StateSize ||
                data.b.Length != agent.ActionSize)
            {
                Debug.LogError(
                    "controller.json has unexpected W/b shapes for ActionSize=" +
                    agent.ActionSize + " StateSize=" + agent.StateSize + ".");
                return false;
            }

            W = data.W;
            b = data.b;
            jointLo = data.joint_lo != null && data.joint_lo.Length == agent.ActionSize
                ? data.joint_lo
                : CopyRangesLo();
            jointHi = data.joint_hi != null && data.joint_hi.Length == agent.ActionSize
                ? data.joint_hi
                : CopyRangesHi();
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            return false;
        }
    }

    float[] CopyRangesLo()
    {
        var result = new float[agent.ActionSize];
        for (int i = 0; i < agent.ActionSize; i++)
            result[i] = goals.jointRanges != null && i < goals.jointRanges.Length
                ? goals.jointRanges[i].x
                : -Mathf.PI;
        return result;
    }

    float[] CopyRangesHi()
    {
        var result = new float[agent.ActionSize];
        for (int i = 0; i < agent.ActionSize; i++)
            result[i] = goals.jointRanges != null && i < goals.jointRanges.Length
                ? goals.jointRanges[i].y
                : Mathf.PI;
        return result;
    }

    [Serializable]
    class ControllerFile
    {
        public string feature;
        public int state_size;
        public int action_size;
        public float[] W;
        public float[] b;
        public float[] joint_lo;
        public float[] joint_hi;
        public float[] home;
    }
}
