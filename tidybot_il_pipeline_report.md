# TidyBot Mobile Manipulator --- IL + RL Pipeline Report

## Overview

This project builds a learning-based control system for a **mobile
manipulator** operating in a robotic cell.\
The robot must navigate to a cell door, position itself correctly, then
use the manipulator to open a sliding door.

Core technologies: - MuJoCo simulator - Gymnasium environment -
Stable-Baselines3 PPO - Imitation Learning (Behavior Cloning + DAgger)

------------------------------------------------------------------------

## Robot System

### Mobile Base

Degrees of freedom: - joint_x - joint_y - joint_th

Holonomic motion enables planar translation and rotation.

### Manipulator

UR‑style arm with 6 joints.\
End‑effector interaction site:

`pinch_site`

------------------------------------------------------------------------

## Environment

Simulator: **MuJoCo**

Environment contains: - mobile manipulator - robotic cell - sliding
door - obstacles - LiDAR sensors - wrist RGB camera

------------------------------------------------------------------------

## Door Model

Door defined in MuJoCo as a sliding joint.

``` xml
<body name="cell_door">
  <joint name="cell_door_joint"
         type="slide"
         axis="-1 0 0"
         range="0 1"/>
</body>
```

Actuation uses a position actuator controlling the slide joint.

Door interaction site:

`cell_door_site`

------------------------------------------------------------------------

## Navigation Philosophy

The base does **not** navigate to a fixed waypoint.

Instead:

`base_target = function(door_site)`

This ensures base placement depends on the manipulation task.

Conceptually:

**The arm carries the base.**

------------------------------------------------------------------------

## Virtual Pre‑Manipulation Pose

The robot navigates to a pose derived from the door site.

Example:

    x_target = door_site_x
    y_target = door_site_y - offset
    yaw_target = facing_door

Parameters:

    prepose_offset_y = 0.75
    prepose_yaw = π/2
    prepose_yaw_tolerance = 0.18

------------------------------------------------------------------------

## Observation Space

Observation dictionary:

    {
     image : wrist camera image
     state : vector
    }

Camera resolution:

128×128 RGB

------------------------------------------------------------------------

## State Vector

    [
     dx_base_prepose,
     dy_base_prepose,
     sin(yaw_error),
     cos(yaw_error),
     distance_to_prepose,
     lidar readings (48),
     hold_counter
    ]

Total dimension:

54

------------------------------------------------------------------------

## Action Space

Continuous action vector:

    [vx, vy, wz]

Normalized range:

    [-1 , 1]

Converted internally to:

    vx = 0.12 m/s
    vy = 0.08 m/s
    wz = 0.35 rad/s

------------------------------------------------------------------------

## Success Condition

Success occurs when:

    distance_to_prepose < 0.25
    yaw_error < 0.18
    hold_steps = 8

Reward bonus:

`success_bonus = 300`

------------------------------------------------------------------------

## Sensors

48 LiDAR rays used for:

-   obstacle detection
-   spawn filtering
-   collision avoidance

Spawn validation rule:

Mean of k smallest lidar distances \> 0.18

------------------------------------------------------------------------

## Imitation Learning Pipeline

Directory structure:

    stanford_tidybot/
      tb_tidybot_nav_il/
        il/

Important files:

-   tidybot_nav_env_v12_stage0.py
-   v12_stage0_config.py
-   teacher.py
-   record_teacher_v12_stage0.py
-   dataset.py
-   nets.py
-   bc_train.py
-   bc_eval_v12_stage0.py
-   dagger_collect_v12_stage0.py

------------------------------------------------------------------------

## Teacher Policy

Hand‑designed proportional controller.

Inputs: - dx - dy - yaw error - distance

Outputs: - vx - vy - wz

Used to generate demonstration datasets.

------------------------------------------------------------------------

## Dataset Recording

Script:

`record_teacher_v12_stage0.py`

Data format:

`.npz shards`

Each shard stores:

-   state
-   action

------------------------------------------------------------------------

## Behavior Cloning

Training script:

`bc_train.py`

Policy: MLP

Input: state vector\
Output: action

------------------------------------------------------------------------

## Evaluation

Script:

`bc_eval_v12_stage0.py`

Metrics: - success rate - distance to goal - yaw error - collisions

------------------------------------------------------------------------

## Current Status

Environment validated.

Example runtime output:

    state shape: (54,)
    lidar len: 48

------------------------------------------------------------------------

## Next Steps

1.  Generate base navigation demonstrations
2.  Train Behavior Cloning policy
3.  Improve using DAgger
4.  Add manipulator IL stage
5.  Train combined base + arm policy
