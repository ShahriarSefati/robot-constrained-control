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

import cvxpy as cp
import numpy as np
import math


class Optimizer:

    def __init__(self, simul, kin):
        self.simul = simul
        self.kin = kin

        # initialize the visualization
        self.q_sim_offset = np.array([math.pi / 2, math.pi / 2, 0, math.pi / 2, 0, math.pi, 0, 0])

        # initialize robot configuration
        self.q = np.array(
            [0.0, -100.0 / 180.0 * math.pi, 110.0 / 180.0 * math.pi, -10.0 / 180.0 * math.pi, 0.0, -math.pi / 2, 0.0,
             0.0])

        self.goal_exists = False
        self.rcm_active = False

        self.epsilon_goal = 0.001  # m
        self.BERNS = True          # Using Bernstein Polynomial for Snake position and Jacobian (True)
                                   # otherwise use regular polynomial of deg 4 (False)

        # initialize constraints matrix and vector
        self.H = np.empty((0, 8), float)
        self.h = np.empty((0), float)

        # constraints' keys
        self.const_keys = []

    def initialize_robot_config(self, config):
        self.q = config

    # add joint position limits virtual fixture
    def add_vf_position_lims(self, qp_l, qp_u):
        self.qp_l = qp_l
        self.qp_u = qp_u
        self.H = np.append(self.H, -np.eye(8), axis=0)
        self.H = np.append(self.H,  np.eye(8), axis=0)
        self.h = np.append(self.h, self.q - self.qp_l, axis=0)
        self.h = np.append(self.h, -self.q + self.qp_u, axis=0)

    # add joint velocity limits virtual fixture
    def add_vf_velocity_lims(self, qv_l, qv_u):
        self.qv_l = qv_l
        self.qv_u = qv_u
        self.H = np.append(self.H, -np.eye(8), axis=0)
        self.H = np.append(self.H,  np.eye(8), axis=0)
        self.h = np.append(self.h, -self.qv_l, axis=0)
        self.h = np.append(self.h, self.qv_u, axis=0)

    # add rcm virtual fixture
    def add_vf_rcm(self, rcm_point, rcm_thresh = 0.001):
        self.rcm_active = True
        self.RCM = rcm_point
        self.epsilon_rcm = rcm_thresh
        self.const_keys.append(['RCM', self.H.shape[0]])
        self.update_kinematics()
        self.H = np.append(self.H, self.V.T * self.cl_jac[0:3, :], axis=0)
        self.h = np.append(self.h, (self.epsilon_rcm + self.V.T * (self.RCM - self.cl_point)).A1, axis=0)


    # add axis range virtual fixture around a desired axis, given the threshold angle
    def add_vf_axis_range(self, axis_des, angle_thresh = 45.0):
        self.axis_des = axis_des
        self.CONE_ANGLE = angle_thresh
        self.const_keys.append(['AXIS', self.H.shape[0]])
        self.update_kinematics()
        self.H = np.append(self.H, -self.axis_des.T * self.cl_jac[3:6, :], axis=0)
        self.h = np.append(self.h, (self.axis_des.T * self.shaft_dir - math.cos(self.CONE_ANGLE / 180.0 * math.pi)).A1,
                           axis=0)

    # add hyper-plane virtual fixture
    # plane eq: ax + by + cz = d
    # => normal=[a, b, c]
    # currently this vf can only be applied to the snake tip or base
    # TODO: the 'base_jac' or 'jac' below should be changed to a general Jacobian resolved at 'point' in inputs
    def add_vf_plane(self, normal, d, snake_tip = True):
        self.update_kinematics()
        if snake_tip:
            self.const_keys.append(['PLANE_TIP', self.H.shape[0], normal, d])
            self.H = np.append(self.H,   normal.T * self.jac[0:3, :], axis=0)
            self.h = np.append(self.h, -(normal.T * self.FK_T[0:3, 3] - d).A1, axis=0)
        else:
            self.const_keys.append(['PLANE_BASE', self.H.shape[0], normal, d])
            self.H = np.append(self.H, -normal.T * self.base_jac[0:3, :], axis=0)
            self.h = np.append(self.h, (normal.T * self.Origin_T[-1][0:3, 3] - d).A1, axis=0)

    # update constraints
    def update_constraints(self):
        for key in self.const_keys:
            if key[0] == 'RCM':
                self.H[key[1]:key[1]+self.V.shape[1],:] = self.V.T * self.cl_jac[0:3, :]
                self.h[key[1]:key[1]+self.V.shape[1]] = (self.epsilon_rcm + self.V.T * (self.RCM - self.cl_point)).A1

            if key[0] == 'AXIS':
                self.H[key[1],:] = -self.axis_des.T * self.cl_jac[3:6, :]
                self.h[key[1]] = (self.axis_des.T * self.shaft_dir - math.cos(self.CONE_ANGLE / 180.0 * math.pi)).A1

            if key[0] == 'PLANE_TIP':
                self.H[key[1], :] = key[2].T * self.jac[0:3, :]
                self.h[key[1]] = -(key[2].T * self.FK_T[0:3, 3] - key[3]).A1

            if key[0] == 'PLANE_BASE':
                self.H[key[1], :] = -key[2].T * self.base_jac[0:3, :]
                self.h[key[1]] = (key[2].T * self.Origin_T[-1][0:3, 3] - key[3]).A1

    # update kinematics
    def update_kinematics(self):
        # forward kinematics
        self.FK_T, self.Origin_T, self.jac, self.snake_pos = self.kin.FK(np.matrix(self.q), self.BERNS)

        if self.rcm_active:
            self.cl_jac, self.cl_point, self.V, self.shaft_dir, self.base_jac = self.kin.closest_point_jac(self.RCM,
                                                                                                       self.Origin_T)

        self.q_snake = self.kin.snake_IK(self.snake_pos)
        self.p_snake = self.Origin_T[-1][0:3, 0:3] * self.kin.snake_kin_from_joints(self.q_snake) + self.Origin_T[-1][
                                                                                                    0:3, 3]

        if self.goal_exists:
            self.x_tip = self.FK_T[0:3, 3].A1
            self.del_x = self.x_goal - self.x_tip

    # Setting objective function
    # Moving toward goal points:  || J del_q - del_x ||
    def objective_func(self):

        self.A = self.jac[0:3, :]
        self.b = self.del_x.T

    ######################
    # constrained solver #
    ######################

    def constrained_opt(self):

        # Define and solve the CVXPY problem.
        del_q = cp.Variable(self.A.shape[1])
        cost = cp.sum_squares(self.A*del_q - self.b)
        prob = cp.Problem(cp.Minimize(cost), [self.H@del_q <= self.h])
        #print(cp.installed_solvers())
        prob.solve(solver=cp.ECOS_BB)
        #prob.solve()

        # Print result.
        #print("Opt: ", prob.status)
        #print("\nThe optimal value is", prob.value)
        #print("The optimal del_q is")
        #print("Del_q: ", del_q.value)
        #print("The norm of the residual is ", cp.norm(A*del_q - b, p=2).value)

        residual = cp.norm(self.A*del_q - self.b, p=2).value

        return del_q.value, prob.status, residual

    def constrained_motion(self, x_goal):

        # initialize goal point
        self.x_goal = x_goal
        self.goal_exists = True

        # updated visualization
        self.simul.set_goal_pos(self.x_goal)

        # update kinematics & constraints
        self.update_kinematics()
        self.update_constraints()

        while True:

            # Checking if the goal is reached
            if np.linalg.norm(self.del_x) < self.epsilon_goal:
                print('Goal point reached.')
                self.simul.set_new_reached_pos(self.x_goal)
                break

            # define objective
            self.objective_func()

            # solve optimization
            del_q, stat, res = self.constrained_opt()

            if stat == 'infeasible':
                print('INFEASIBLE problem; the optimization constraints are too tight.\n')
                break

            # update kinematics & constraints
            self.q = self.q + del_q
            self.update_kinematics()
            self.update_constraints()

            # update visualization
            self.simul.set_motor_joint_pos(self.q + self.q_sim_offset)
            self.simul.set_tip_pos(self.x_tip)
            self.simul.set_snakeJoint_pos(self.p_snake)

            if self.rcm_active:
                self.simul.set_rcm_pos(self.RCM)
