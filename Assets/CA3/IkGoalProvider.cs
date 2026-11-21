using UnityEngine;
using Mujoco;

// Cartesian IK that produces joint goals for the learned controller.
// Method: damped least squares (DLS) on the position Jacobian, one solve per arm per physics step.
// [AI-assisted] Execution order -20 so the goal is set before ArmJointController (-10) and
// WorldModelController (-5) read it in the same physics step.
[DefaultExecutionOrder(-20)]
public class IkGoalProvider : MonoBehaviour
{
    public ArmAgent armAgent;
    public Transform lefttarget;
    public Transform righttarget;
    public Transform leftgoal;
    public Transform rightgoal;

    public int iterations = 10;
    public float stepSize = 0.5f;
    public float damping = 0.05f;
    public float gripperOpen = 0.02f;

    // [AI-assisted] Added: the controller that receives the joint goals, and a small pull towards
    // the home pose in the Jacobian null space so the redundant wrist DOFs do not drift.
    public ArmJointController goals;
    public float nullSpaceGain = 0.05f;
    public float tolerance = 0.001f;   // stop iterating when the end effector is within 1 mm

    // [AI-assisted] IK point = the MJCF grasp sites between the fingertips ("left/gripper",
    // "right/gripper", 13 cm in front of gripper_link). Using the gripper_link body origin
    // (the wrist) made the IK push the wrist onto the target, swallowing the ball.
    public MjSite leftTip;
    public MjSite rightTip;
    public string leftTipName = "left/gripper";
    public string rightTipName = "right/gripper";

    float[] leftGoal;
    float[] rightGoal;
    double[] savedQ;
    bool ready;

    // [AI-assisted] Added buffers.
    const int ArmJoints = 6;           // hinge joints per arm (gripper handled separately)
    const int ArmStride = 7;           // per arm in ArmAgent order: 6 hinges + gripper
    float[] goal;                      // all 14 joint goals passed to ArmJointController.SetTarget
    double[] jacp;                     // 3 x nv position Jacobian from MuJoCo
    readonly double[,] J = new double[3, ArmJoints];

    // [AI-assisted] Rewritten: ArmAgent has no leftArmJointCount / totalJointCount / model / data;
    // sizes come from ArmAgent.ActionSize / JointCount and MuJoCo data is read from MjScene.
    void Start()
    {
        if (armAgent == null)
            armAgent = GetComponent<ArmAgent>();
        if (goals == null)
            goals = GetComponent<ArmJointController>();
        if (leftTip == null)
            leftTip = FindSite(leftTipName);
        if (rightTip == null)
            rightTip = FindSite(rightTipName);
        if (armAgent == null || !armAgent.Ready || goals == null ||
            lefttarget == null || righttarget == null || leftTip == null || rightTip == null)
        {
            Debug.LogError("IkGoalProvider needs a ready ArmAgent, an ArmJointController and both targets.");
            enabled = false;
            return;
        }

        leftGoal = new float[ArmStride];
        rightGoal = new float[ArmStride];
        savedQ = new double[armAgent.JointCount];
        goal = new float[armAgent.ActionSize];
        // The warm start is copied in FixedUpdate: this Start runs before ArmJointController.Start
        // (execution order -20 vs -10), so its target array does not exist yet.
        ready = true;
    }

    bool warmStarted;

    // [AI-assisted] Find an MjSite by its MJCF name.
    static MjSite FindSite(string siteName)
    {
        foreach (var site in FindObjectsByType<MjSite>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            if (site.name == siteName)
                return site;
        Debug.LogWarning("IkGoalProvider: site '" + siteName + "' not found.");
        return null;
    }

    // [AI-assisted] Rewritten: runs in FixedUpdate (in step with MuJoCo) instead of Update, and uses
    // the save -> overwrite qpos for FK -> restore pattern so the real simulation is untouched.
    unsafe void FixedUpdate()
    {
        if (!ready || !armAgent.Ready || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;

        var model = MjScene.Instance.Model;
        var data = MjScene.Instance.Data;
        if (jacp == null)
            jacp = new double[3 * (int)model->nv];

        // Warm start from the current joint goal (home pose at startup).
        if (!warmStarted)
        {
            if (goals.CurrentTarget == null)
                return;
            System.Array.Copy(goals.CurrentTarget, goal, goal.Length);
            warmStarted = true;
        }

        // 1. Save the real joint positions.
        for (int i = 0; i < armAgent.JointCount; i++)
            savedQ[i] = data->qpos[armAgent.joints[i].QposAddress];

        // 2. Solve each arm independently (left = joints 0..5, right = joints 7..12).
        Vector3 leftSolved = SolveArm(model, data, 0, leftTip, lefttarget);
        Vector3 rightSolved = SolveArm(model, data, ArmStride, rightTip, righttarget);
        goal[ArmStride - 1] = gripperOpen;          // left gripper
        goal[2 * ArmStride - 1] = gripperOpen;      // right gripper

        // 3. Restore the real joint positions and recompute FK (otherwise the real arm teleports).
        for (int i = 0; i < armAgent.JointCount; i++)
            data->qpos[armAgent.joints[i].QposAddress] = savedQ[i];
        MujocoLib.mj_kinematics(model, data);

        // 4. Hand the joint goals to the controller (kept in leftGoal/rightGoal for inspection).
        System.Array.Copy(goal, 0, leftGoal, 0, ArmStride);
        System.Array.Copy(goal, ArmStride, rightGoal, 0, ArmStride);
        goals.SetTarget(goal);

        // Optional markers: where the IK solution puts each hand (differs from the target when
        // the target is out of reach).
        if (leftgoal != null)
            leftgoal.position = leftSolved;
        if (rightgoal != null)
            rightgoal.position = rightSolved;
    }

    // [AI-assisted] Completed: damped least squares IK for one arm.
    //   e  = target - p(q)                       (3)
    //   dq = J^T (J J^T + lambda^2 I)^-1 e       (6)
    //   dq += (I - J^+ J) k (home - q)           (null-space pull towards home)
    //   q  = clamp(q + alpha * dq, joint limits)
    // Returns the solved end-effector position in Unity coordinates.
    unsafe Vector3 SolveArm(MujocoLib.mjModel_* model, MujocoLib.mjData_* data,
                            int first, MjSite endEffector, Transform target)
    {
        if (endEffector == null || target == null)
            return Vector3.zero;

        // Unity (x, y, z) -> MuJoCo (x, z, y): MuJoCo is z-up, Unity is y-up.
        Vector3 t = target.position;
        double tx = t.x, ty = t.z, tz = t.y;
        int site = endEffector.MujocoId;
        int nv = (int)model->nv;
        double lambda2 = damping * damping;

        var dq = new double[ArmJoints];
        var y = new double[3];
        var e = new double[3];

        for (int it = 0; it < iterations; it++)
        {
            // Forward kinematics at the current guess for this arm.
            for (int c = 0; c < ArmJoints; c++)
                data->qpos[armAgent.joints[first + c].QposAddress] = goal[first + c];
            MujocoLib.mj_kinematics(model, data);

            e[0] = tx - data->site_xpos[3 * site + 0];
            e[1] = ty - data->site_xpos[3 * site + 1];
            e[2] = tz - data->site_xpos[3 * site + 2];
            if (e[0] * e[0] + e[1] * e[1] + e[2] * e[2] < tolerance * tolerance)
                break;

            // Position Jacobian (3 x nv), keep only this arm's 6 hinge columns.
            MujocoLib.mj_comPos(model, data);
            fixed (double* jp = jacp)
                MujocoLib.mj_jacSite(model, data, jp, null, site);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < ArmJoints; c++)
                    J[r, c] = jacp[r * nv + armAgent.joints[first + c].DofAddress];

            // A = J J^T + lambda^2 I  (3 x 3), solve A y = e, dq = J^T y.
            var A = new double[3, 3];
            for (int r = 0; r < 3; r++)
                for (int s = 0; s < 3; s++)
                {
                    double sum = 0;
                    for (int c = 0; c < ArmJoints; c++)
                        sum += J[r, c] * J[s, c];
                    A[r, s] = sum + (r == s ? lambda2 : 0);
                }
            if (!Solve3(A, e, y))
                break;
            for (int c = 0; c < ArmJoints; c++)
                dq[c] = J[0, c] * y[0] + J[1, c] * y[1] + J[2, c] * y[2];

            // Null-space term: z = k (home - q); dq += z - J^T A^-1 J z.
            if (nullSpaceGain > 0f)
            {
                var z = new double[ArmJoints];
                var jz = new double[3];
                for (int c = 0; c < ArmJoints; c++)
                    z[c] = nullSpaceGain * (goals.homeTargets[first + c] - goal[first + c]);
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < ArmJoints; c++)
                        jz[r] += J[r, c] * z[c];
                var w = new double[3];
                if (Solve3(A, jz, w))
                    for (int c = 0; c < ArmJoints; c++)
                        dq[c] += z[c] - (J[0, c] * w[0] + J[1, c] * w[1] + J[2, c] * w[2]);
            }

            for (int c = 0; c < ArmJoints; c++)
            {
                int j = first + c;
                goal[j] = Mathf.Clamp(goal[j] + stepSize * (float)dq[c],
                                      goals.jointRanges[j].x, goals.jointRanges[j].y);
            }
        }

        // FK at the final solution, converted back to Unity (x, z, y).
        for (int c = 0; c < ArmJoints; c++)
            data->qpos[armAgent.joints[first + c].QposAddress] = goal[first + c];
        MujocoLib.mj_kinematics(model, data);
        return new Vector3((float)data->site_xpos[3 * site + 0],
                           (float)data->site_xpos[3 * site + 2],
                           (float)data->site_xpos[3 * site + 1]);
    }

    // [AI-assisted] Added: solve a 3x3 linear system with Cramer's rule. Returns false if singular.
    static bool Solve3(double[,] a, double[] b, double[] x)
    {
        double det =
            a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]) -
            a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0]) +
            a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]);
        if (System.Math.Abs(det) < 1e-12)
            return false;
        double inv = 1.0 / det;
        x[0] = inv * (b[0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]) -
                      a[0, 1] * (b[1] * a[2, 2] - a[1, 2] * b[2]) +
                      a[0, 2] * (b[1] * a[2, 1] - a[1, 1] * b[2]));
        x[1] = inv * (a[0, 0] * (b[1] * a[2, 2] - a[1, 2] * b[2]) -
                      b[0] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0]) +
                      a[0, 2] * (a[1, 0] * b[2] - b[1] * a[2, 0]));
        x[2] = inv * (a[0, 0] * (a[1, 1] * b[2] - b[1] * a[2, 1]) -
                      a[0, 1] * (a[1, 0] * b[2] - b[1] * a[2, 0]) +
                      b[0] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]));
        return true;
    }
}
