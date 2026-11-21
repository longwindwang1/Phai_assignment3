using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

public class RolloutRecorder : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController controller;
    public int steps = 70000;
    public int logEvery = 500;

    readonly List<float[]> states = new();
    readonly List<float[]> actions = new();
    readonly List<float[]> nextStates = new();

    int n;
    float[] prevS;
    float[] prevA;
    bool saved;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (controller == null)
            controller = GetComponent<ArmJointController>();
        if (agent == null || !agent.Ready || controller == null)
        {
            Debug.LogError("RolloutRecorder needs a ready ArmAgent and ArmJointController.");
            enabled = false;
            return;
        }
    }

    void FixedUpdate()
    {
        if (saved || !agent.Ready)
            return;

        // Copy, never keep references: Observe()/CurrentAction reuse the same arrays every step.
        var state = new float[agent.StateSize];
        agent.CopyState(state);
        var action = (float[])agent.CurrentAction.Clone();

        // (state before the step, action applied during it, state after it)
        if (prevS != null && prevA != null)
        {
            states.Add(prevS);
            actions.Add(prevA);
            nextStates.Add(state);
            n++;

            if (logEvery > 0 && n % logEvery == 0)
                Debug.Log($"RolloutRecorder: {n}/{steps} steps, tracking error {TrackingError():F4}");
        }

        prevS = state;
        prevA = action;

        if (n >= steps)
        {
            saved = true;
            Save();
        }
    }

    float TrackingError()
    {
        // RMS of (joint position - current goal); only the first ActionSize entries are positions.
        var state = new float[agent.StateSize];
        agent.CopyState(state);
        float[] target = controller.CurrentTarget;
        if (target == null)
            return 0f;
        float sum = 0f;
        for (int i = 0; i < agent.ActionSize; i++)
        {
            float error = state[i] - target[i];
            sum += error * error;
        }
        return Mathf.Sqrt(sum / agent.ActionSize);
    }

    void Save()
    {
        // Data/rollout.csv next to train.py (Assets/CA3/Data), headerless: q, qd, q*, q', qd'
        string dir = Path.Combine(Application.dataPath, "CA3", "Data");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "rollout.csv");

        var sb = new StringBuilder();
        var inv = CultureInfo.InvariantCulture;
        for (int i = 0; i < states.Count; i++)
        {
            AppendRow(sb, states[i], inv);
            sb.Append(',');
            AppendRow(sb, actions[i], inv);
            sb.Append(',');
            AppendRow(sb, nextStates[i], inv);
            sb.Append('\n');
        }
        File.WriteAllText(path, sb.ToString());
        Debug.Log($"RolloutRecorder: saved {states.Count} rows to {path}");
    }

    static void AppendRow(StringBuilder sb, float[] values, CultureInfo inv)
    {
        for (int j = 0; j < values.Length; j++)
        {
            if (j > 0)
                sb.Append(',');
            sb.Append(values[j].ToString("R", inv));
        }
    }
}
