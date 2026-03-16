from pathlib import Path
import math

from pathlib import Path
import math

try:
    from tb_tidybot_nav_il.il.tidybot_nav_env_v12_stage0 import V12Stage0Params
except ImportError:
    from tidybot_nav_env_v12_stage0 import V12Stage0Params
# Resolve XML relative to project root:
# stanford_tidybot/tb_tidybot_nav_il/il/v12_stage0_config.py
# -> parents[2] = stanford_tidybot
ROOT = Path(__file__).resolve().parents[2]

ENV = V12Stage0Params(
    xml_path=str(ROOT / "tidybot_with_cell.xml"),

    ee_site_name="pinch_site",
    target_site_name="cell_door_site",
    camera_name="wrist",
    robot_root_body="base_link",

    dt=0.05,
    physics_dt=0.001,
    max_steps=700,

    cam_w=128,
    cam_h=128,

    max_vx=0.12,
    max_vy=0.08,
    max_wz=0.35,

    spawn_xy_range=0.5,
    spawn_yaw_range=math.pi,

    spawn_around_target=False,
    spawn_target_min_r=0.35,
    spawn_target_max_r=0.65,

    # manipulation-conditioned virtual prepose
    prepose_offset_y=0.75,
    prepose_yaw=math.pi / 2.0,
    prepose_yaw_tolerance=0.18,

    base_success_radius=0.25,
    hold_steps=8,

    tau=0.25,

    w_prog_anchor=10.0,
    w_dist_anchor=0.0,
    w_time=0.01,
    w_ctrl_base=0.01,
    success_bonus=300.0,

    terminate_on_collision=True,
    collision_penalty=80.0,
    collision_grace_radius=1.2,

    lidar_max_range=4.0,
    spawn_reject_using_lidar=True,
    spawn_k_smallest=6,
    spawn_min_kmean_lidar=0.18,
    spawn_max_tries=60,
    settle_steps=30,

    base_joint_x="joint_x",
    base_joint_y="joint_y",
    base_joint_th="joint_th",
    base_act_x="joint_x",
    base_act_y="joint_y",
    base_act_th="joint_th",
)