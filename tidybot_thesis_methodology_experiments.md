# Thesis Methodology and Experiment Plan

## Learning-Based Mobile Manipulator Door Interaction

### Research Objective

Develop a learning-based system allowing a mobile manipulator to
autonomously:

1.  Navigate to a robotic cell
2.  Position itself for manipulation
3.  Reach a door handle
4.  Open a sliding door

The research focuses on **learning coordinated base--arm behavior**.

------------------------------------------------------------------------

## System Architecture

The system integrates:

-   MuJoCo physics simulation
-   Gymnasium reinforcement learning environments
-   Stable-Baselines3 PPO
-   Imitation Learning (Behavior Cloning + DAgger)

Training proceeds in stages.

------------------------------------------------------------------------

## Stage 1 --- Base Navigation

Goal: Learn to navigate the mobile base to a manipulation-ready pose.

State representation:

-   relative base position to door site
-   orientation error
-   LiDAR obstacle distances

Action:

    [vx, vy, wz]

Learning approach:

-   imitation learning from a teacher controller
-   Behavior Cloning

------------------------------------------------------------------------

## Stage 2 --- Policy Refinement

Use **DAgger (Dataset Aggregation)**.

Purpose:

-   reduce compounding errors
-   improve policy robustness

Procedure:

1.  Run learned policy
2.  Query teacher for corrective actions
3.  Aggregate dataset
4.  retrain policy

------------------------------------------------------------------------

## Stage 3 --- Manipulator Learning

After base navigation stabilizes:

Train manipulator to:

-   reach door handle
-   apply force along sliding axis

Inputs:

-   end-effector position
-   door handle site location
-   camera observation

Outputs:

-   arm joint commands

------------------------------------------------------------------------

## Stage 4 --- Coordinated Control

Combine base and arm control into a single policy.

The robot must:

-   move base to correct location
-   activate manipulator
-   perform door-opening motion

------------------------------------------------------------------------

## Experiment Metrics

Experiments evaluate:

1.  Navigation success rate
2.  Manipulation success rate
3.  Task completion time
4.  Collision frequency

------------------------------------------------------------------------

## Expected Contributions

The thesis aims to demonstrate:

-   learning-based navigation for manipulation
-   coordination between mobile base and robotic arm
-   imitation learning for mobile manipulation tasks

------------------------------------------------------------------------

## Final Demonstration

The trained policy performs:

1.  approach robotic cell
2.  align with door
3.  reach handle
4.  open sliding door

This demonstrates **mobile manipulator collaboration through
learning-based control**.
