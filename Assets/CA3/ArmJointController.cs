using UnityEngine;


[DefaultExecutionOrder(-10)]
public class ArmJointController : MonoBehaviour
{
    public ArmAgent agent;
    public float holdSeconds = 3f;

    public Vector2[] jointRanges;

    public float[] homeTargets;

    [Range(0f, 1f)]
    public float randomScale = 0.35f;

    public bool emitActions = true;

    // Keep sampling random goals every holdSeconds even when emitActions is false.
    // Used to verify the learned controller on the data-collection goals; turn off when
    // goals come from IK instead.
    public bool randomizeGoals = true;

    float[] target;
    float hold;
    bool ready;

    public float[] CurrentTarget => target;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (agent == null || !agent.Ready)
        {
            Debug.LogError("ArmJointController needs a ready ArmAgent.");
            enabled = false;
            return;
        }
        if (homeTargets == null || homeTargets.Length != agent.ActionSize ||
            jointRanges == null || jointRanges.Length != agent.ActionSize)
        {
            Debug.LogError(
                "ArmJointController: homeTargets and jointRanges must both have " +
                agent.ActionSize + " entries.");
            enabled = false;
            return;
        }

        target = new float[agent.ActionSize];
        PlaceTarget(true);
        hold = 0f;
        ready = true;
    }

    void FixedUpdate()
    {
        if (!ready)
            return;

        // Pick a new random goal every holdSeconds while collecting data (emitActions) or while
        // verifying the learned controller on the same random goals (randomizeGoals).
        if (emitActions || randomizeGoals)
        {
            hold += Time.fixedDeltaTime;
            if (hold >= holdSeconds)
            {
                hold = 0f;
                PlaceTarget(false);
            }
        }

        // Only drive the actuators directly when no controller has taken over.
        if (emitActions)
            agent.SetAction(target);
    }

    // Lets IK (or anything else) set the joint goal that the learned controller tracks.
    public void SetTarget(float[] goal)
    {
        if (!ready || goal == null)
            return;
        int n = Mathf.Min(goal.Length, target.Length);
        for (int i = 0; i < n; i++)
            target[i] = Mathf.Clamp(goal[i], jointRanges[i].x, jointRanges[i].y);
    }

    public void ActionFrom(float[] destination)
    {
        if (!ready || destination == null)
            return;
        int n = Mathf.Min(destination.Length, target.Length);
        for (int i = 0; i < n; i++)
            destination[i] = target[i];
    }

    public void PlaceTarget(bool home)
    {
        if (target == null)
            return;

        for (int i = 0; i < target.Length; i++)
        {
            float lo = jointRanges[i].x;
            float hi = jointRanges[i].y;
            if (home)
            {
                target[i] = Mathf.Clamp(homeTargets[i], lo, hi);
            }
            else
            {
                // Same distribution as train_controller.sample_goal:
                // home ± (hi - lo) / 2 * randomScale, clipped to the joint range.
                float span = (hi - lo) * 0.5f * randomScale;
                target[i] = Mathf.Clamp(homeTargets[i] + Random.Range(-span, span), lo, hi);
            }
        }
    }
}
