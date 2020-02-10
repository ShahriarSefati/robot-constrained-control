#!/usr/bin/env python
"""

Copyright (c) 2019-2020 Shahriar (Yar) Sefati, Johns Hopkins University

@author Shahriar Sefati <shahriar.sefati@gmail.com>


BSD license:
------------

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np
import math

from cmc_kinematics import Kinematics
from cmc_simulation import Simulation
from cmc_optimizer import Optimizer

# initalize objects
simul = Simulation()
kin = Kinematics('EXTENDED-UR5')
optimizer = Optimizer(simul, kin)

# Axis range constraint (cone)
CONE_ANGLE = 45.0 # degree

# Plane constraints
PENETRATION_OFFSET = 0.052 # m
BASE_OFFSET = 0.015 # m

# Joint position limits
ROBOT_POS_LIM = 2 * math.pi # rad
ROLL_POS_LIM = 200 * math.pi # rad
SNAKE_POS_LIM = 0.010 # mm
qp_l = np.array([-ROBOT_POS_LIM, -ROBOT_POS_LIM, -ROBOT_POS_LIM, -ROBOT_POS_LIM,
                 -ROBOT_POS_LIM, -ROBOT_POS_LIM, -ROLL_POS_LIM, -SNAKE_POS_LIM])
qp_u = np.array([ROBOT_POS_LIM, ROBOT_POS_LIM, ROBOT_POS_LIM, ROBOT_POS_LIM,
                 ROBOT_POS_LIM, ROBOT_POS_LIM, ROLL_POS_LIM, SNAKE_POS_LIM])

# Joint velocity limits
ROBOT_VEL_LIM = 0.005 # rad/s
ROLL_VEL_LIM = 0.05 # rad/s
SNAKE_VEL_LIM = 0.0002 # mm/s

qv_l = np.array([-ROBOT_VEL_LIM, -ROBOT_VEL_LIM, -ROBOT_VEL_LIM, -ROBOT_VEL_LIM, -ROBOT_VEL_LIM, -ROBOT_VEL_LIM, -ROLL_VEL_LIM, -SNAKE_VEL_LIM])
qv_u = np.array([ ROBOT_VEL_LIM,  ROBOT_VEL_LIM,  ROBOT_VEL_LIM,  ROBOT_VEL_LIM,  ROBOT_VEL_LIM,  ROBOT_VEL_LIM,  ROLL_VEL_LIM,  SNAKE_VEL_LIM])

# initialize robot configuration
q_init = np.array([0.0, -100.0/180.0 * math.pi, 110.0/180.0 *math.pi, -10.0/180.0 * math.pi, 0.0, -math.pi/2, 0.0, 0.0])  # Closer to robot eef


# Initialize positions and configurations
BERNS =True
FK_T, Origin_T, jac, snake_pos = kin.FK(np.matrix(q_init), BERNS)
RCM = Origin_T[-1][0:3, 3]
BASE = Origin_T[-1][0:3, 3]
x_tip = FK_T[0:3, 3].A1  # A1 makes the matrix an array
cl_jac, cl_point, V, axis_des, base_jac = kin.closest_point_jac(RCM, Origin_T)

# Start simulation connection and get object handles
simul.start_connection()
simul.get_handles()

# generate a set of goal points
goal_list = []
n = 20
for i in range(n):
    goal_list.append(np.array([x_tip[0], x_tip[1] , x_tip[2]+ i * 0.002]))

print('Number of target goal points: ', len(goal_list))


# Optimization Problem
# Here you can add as many constraints as you like
optimizer.initialize_robot_config(q_init)                                    # Initialize robot configuration
optimizer.add_vf_position_lims(qp_l, qp_u)                                   # Joint position limit constraint
optimizer.add_vf_velocity_lims(qv_l, qv_u)                                   # Joint velocity limit constraint
optimizer.add_vf_rcm(RCM, 0.001)                                             # RCM constraint
optimizer.add_vf_axis_range(axis_des, 30.0)                                  # Axis range constraint
optimizer.add_vf_plane(axis_des, axis_des.T *
                    (BASE - BASE_OFFSET * axis_des), snake_tip = False)       # Keep snake base from pulling back too far
optimizer.add_vf_plane(axis_des, axis_des.T *
                    (BASE + PENETRATION_OFFSET * axis_des), snake_tip = True) # Keep snake tip from penetrating too far


# go through the list of goal points
for i in range(len(goal_list)):
    optimizer.constrained_motion(goal_list[i])
