import numpy as np
from utils import mktransf, quaternion2yaw

# columns used as input for the model
input_cols = ['pos_x', 'pos_y', 'theta',
              # note: this is used for the simulated thymio
              #   'ooi_pos_x', 'ooi_pos_y', 'ooi_theta',
              'camera']

# columns used as target for the model
target_cols = ['target_left_sensor',
               'target_center_left_sensor',
               'target_center_sensor',
               'target_center_right_sensor',
               'target_right_sensor']

# a list of coordinates in a regular grid, shape = (n, 2)
coords = np.stack(np.meshgrid(
    np.linspace(0, .8, int(.8 / .04)),
    np.linspace(-.4, .4, int(.8 / .04))
)).reshape([2, -1]).T

# transformation matrices of sensors' frames w.r.t. robot frame (for the thymio)
robot_geometry = [
    mktransf((0.0630, 0.0493,
              quaternion2yaw([0.0, 0.0, 0.3256, 0.9455]))),  # left
    mktransf((0.0756, 0.0261,
              quaternion2yaw([0.0, 0.0, 0.1650, 0.9863]))),  # center_left
    mktransf((0.0800, 0.0000, 0.0000)),  # center
    mktransf((0.0756, -0.0261,
              quaternion2yaw([0.0, 0.0, -0.1650, 0.9863]))),  # center_right
    mktransf((0.0630, -0.0493,
              quaternion2yaw([0.0, 0.0, -0.3256, 0.9455])))  # right
]

# left_rot = mkrot(quaternion2yaw([0.0, 0.0, 0.3256, 0.9455]))
# left_trs = mktr(0.063, 0.0493)

# center_left_rot = mkrot(quaternion2yaw([0.0, 0.0, 0.1650, 0.9863]))
# center_left_trs = mktr(0.0756, 0.0261)

# center_rot = mkrot(quaternion2yaw([0.0, 0.0, 0.0, 1.0]))
# center_trs = mktr(0.08, 0)

# center_right_rot = mkrot(quaternion2yaw([0.0, 0.0, -0.1650, 0.9863]))
# center_right_trs = mktr(0.0756, -0.0261)

# right_rot = mkrot(quaternion2yaw([0.0, 0.0, -0.3256, 0.9455]))
# right_trs = mktr(0.063, -0.0493)
