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

import sim


class Simulation:

    def __init__(self):
        self.jointNames = ['UR5_joint1', 'UR5_joint2', 'UR5_joint3', 'UR5_joint4', 'UR5_joint5', 'UR5_joint6']
        self.jointHandles = [0, 0, 0, 0, 0, 0]
        self.objNames = ['RCM', 'GOAL', 'TIP', 'CL', 'CUP','REACHED', 'PLANE', 'CONE', 'ELLIPSOID', 'TEST']
        self.objHandles = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.rcmPlaneNames = ['RCM_PL1', 'RCM_PL2', 'RCM_PL3', 'RCM_PL4', 'RCM_PL5','RCM_PL6', 'RCM_PL7', 'RCM_PL8']
        self.rcmPlaneHandles = [0, 0, 0, 0, 0, 0, 0, 0]
        self.FKjointNames = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8']
        self.FKjointHandles = [0, 0, 0, 0, 0, 0, 0, 0]
        self.snakeJointNames = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8',
                                'S9', 'S10', 'S11', 'S12','S13', 'S14', 'S15',
                                'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22',
                                'S23', 'S24', 'S25', 'S26', 'S27']
        self.snakeJointHandles = [0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0]

    ##############
    # Connection #
    ##############

    def start_connection(self):
        print('Connecting to server...')
        sim.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim
        if self.clientID != -1:
            print('Connected to remote API server')
        else:
            print('Failed to connect to remote API server')
            exit(-1)

    ###############
    # Get Methods #
    ###############

    def get_handles(self):
        self.get_motor_handles()
        self.get_FKjoint_handles()
        self.get_snakeJoint_handles()
        self.get_obj_handles()
        self.get_rcm_plane_handles()

    def get_motor_handles(self):
        for i in range(len(self.jointNames)):
            err_code, self.jointHandles[i] = sim.simxGetObjectHandle(self.clientID, self.jointNames[i], sim.simx_opmode_blocking)

    def get_FKjoint_handles(self):
        for i in range(len(self.FKjointNames)):
            err_code, self.FKjointHandles[i] = sim.simxGetObjectHandle(self.clientID, self.FKjointNames[i], sim.simx_opmode_blocking)

    def get_snakeJoint_handles(self):
        for i in range(len(self.snakeJointNames)):
            err_code, self.snakeJointHandles[i] = sim.simxGetObjectHandle(self.clientID, self.snakeJointNames[i], sim.simx_opmode_blocking)

    def get_obj_handles(self):
        for i in range(len(self.objNames)):
            err_code, self.objHandles[i] = sim.simxGetObjectHandle(self.clientID, self.objNames[i],
                                                                 sim.simx_opmode_blocking)

    def get_rcm_plane_handles(self):
        for i in range(len(self.rcmPlaneNames)):
            err_code, self.rcmPlaneHandles[i] = sim.simxGetObjectHandle(self.clientID, self.rcmPlaneNames[i],
                                                                 sim.simx_opmode_blocking)


    ###############
    # Set Methods #
    ###############

    def set_motor_joint_pos(self, q):
        for i in range(len(self.jointNames)):
            err_code = sim.simxSetJointTargetPosition(self.clientID,self.jointHandles[i], q[i], sim.simx_opmode_streaming)

    def set_FKjoint_pos(self, Origin_T):
        for i in range(len(self.FKjointNames)):
            err_code = sim.simxSetObjectPosition(self.clientID,self.FKjointHandles[i], -1, Origin_T[i][0:3,3], sim.simx_opmode_streaming)

    def set_snakeJoint_pos(self, snake_pos):
        for i in range(len(self.snakeJointNames)):
            err_code = sim.simxSetObjectPosition(self.clientID,self.snakeJointHandles[i], -1, snake_pos[:,i], sim.simx_opmode_streaming)

    def set_rcm_pos(self, pos):
        err_code = sim.simxSetObjectPosition(self.clientID, self.objHandles[0], -1, pos, sim.simx_opmode_streaming)

    def set_cl_pos(self, pos):
        err_code = sim.simxSetObjectPosition(self.clientID, self.objHandles[3], -1, pos, sim.simx_opmode_streaming)

    def set_goal_pos(self, pos):
        err_code = sim.simxSetObjectPosition(self.clientID, self.objHandles[1], -1, pos, sim.simx_opmode_streaming)

    def set_tip_pos(self, pos):
        err_code = sim.simxSetObjectPosition(self.clientID, self.objHandles[2], -1, pos, sim.simx_opmode_streaming)

    def set_new_reached_pos(self, pos):
        err_code, point_handle = sim.simxCopyPasteObjects(self.clientID, [self.objHandles[5]], sim.simx_opmode_blocking)
        err_code = sim.simxSetObjectPosition(self.clientID, point_handle[0], -1, pos, sim.simx_opmode_streaming)