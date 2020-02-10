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
from math import cos as cos
from math import sin as sin
from math import pi as pi
from scipy.optimize import minimize
from scipy.optimize import Bounds

mat = np.matrix


class Kinematics:

    def __init__(self, robot_type):
        if robot_type == 'UR5':
            self.d = mat([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])            # UR5
            self.a = mat([0, -0.425, -0.39225, 0, 0, 0])                        # UR5
            self.alph = mat([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])  # UR5

        if robot_type == 'UR10':
            self.d = mat([0.1273, 0, 0, 0.163941, 0.1157, 0.0922])              # UR10
            self.a = mat([0, -0.612, -0.5723, 0, 0, 0])                         # UR10
            self.alph = mat([pi / 2, 0, 0, pi / 2, -pi / 2, 0])                 # UR10

        if robot_type == 'EXTENDED-UR5':
            self.d = mat([0.089159, 0, 0, 0.10915, 0.09465, 0.0823 + 0.0750998, 0.37980142])    #extended-UR5 (+Roll)
            self.a =mat([0, -0.425, -0.39225, 0, 0, 0, 0])                                      #extended-UR5 (+Roll)
            self.alph = mat([math.pi/2, 0, 0, math.pi/2, -math.pi/2, math.pi/2, 0])            #extended-UR5 (+Roll)

    def AH(self, n, th):
        T_a = mat(np.identity(4), copy=False)
        T_a[0, 3] = self.a[0, n]
        T_d = mat(np.identity(4), copy=False)
        T_d[2, 3] = self.d[0, n]

        Rzt = mat([[cos(th[0, n]), -sin(th[0, n]), 0, 0],
                [sin(th[0, n]), cos(th[0, n]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], copy=False)

        Rxa = mat([[1, 0, 0, 0],
                [0, cos(self.alph[0, n]), -sin(self.alph[0, n]), 0],
                [0, sin(self.alph[0, n]), cos(self.alph[0, n]), 0],
                [0, 0, 0, 1]], copy=False)

        A_i = T_d * Rzt * T_a * Rxa

        return A_i

    def FK(self, th, BERNS):
        A = []
        # index 0 to 6 for UR and Roll
        for i in range(0,7):
            A.append(self.AH(i, th))

        # index 7 for snake cable
        A_snake_pos = self.snake_kin(th[0,7], BERNS)
        A.append(mat([ [A[6][0,0], A[6][0,1], A[6][0,2], A_snake_pos[0,0]],
                    [A[6][1,0], A[6][1,1], A[6][1,2], A_snake_pos[0,1]],
                    [A[6][2,0], A[6][2,1], A[6][2,2], A_snake_pos[0,2]],
                    [0, 0, 0, 1]]))

        FK_T = mat(np.eye(4))  # Starting the transformations
        Origins_T = []         # Storing the transformations from base to each joint

        # compute FK
        for j in range(0,8):
            Origins_T.append(FK_T) # first origin is the identity (joint 1's frame)
            FK_T = FK_T * A[j]


        # compute Jacobian
        jac = mat(np.eye(6,8))
        tip = FK_T[0:3, 3]

        for k in range(0,7):
            z_joint = Origins_T[k][0:3, 2] # 3rd column of the rotation part
            o_joint = Origins_T[k][0:3, 3] # 4th column of the transformation (translation component)

            jac[0:3, k] = mat(np.cross(z_joint.A1, (tip - o_joint).A1)).T
            jac[3:6, k] = z_joint

        # snake jac
        jac[0:3, 7] = Origins_T[6][0:3, 0:3] * self.snake_jac(th[0,7], BERNS).T # Rotation of the base of the snake multiplied by snake jac
        jac[3:6, 7] = mat([0.0, 0.0, 0.0]).T

        return FK_T, Origins_T, jac, A_snake_pos.A1

    def find_cl_point_on_line(self, RCM, line_pt, normalized_dir):
        cl = line_pt + normalized_dir * np.dot((RCM - line_pt).A1, normalized_dir.A1)

        return cl

    def closest_point_jac(self, RCM, Origins_T):
        line_pt = Origins_T[7][0:3,3]
        line_shaft_dir = Origins_T[7][0:3,3] - Origins_T[6][0:3,3]

        normalized_shaft_dir = line_shaft_dir / np.linalg.norm(line_shaft_dir)

        cl_point = self.find_cl_point_on_line(RCM, line_pt, normalized_shaft_dir)
        snake_base = Origins_T[-1][0:3,3]

        cl_jac = mat(np.zeros((6,8)))
        base_jac = mat(np.zeros((6,8)))

        for k in range(0,7):
            z_joint = Origins_T[k][0:3, 2] # 3rd column of the rotation part
            o_joint = Origins_T[k][0:3, 3] # 4th column of the transformation (translation component)

            cl_jac[0:3, k] = mat(np.cross(z_joint.A1, (cl_point - o_joint).A1)).T
            cl_jac[3:6, k] = z_joint

            base_jac[0:3, k] = mat(np.cross(z_joint.A1, (snake_base - o_joint).A1)).T
            base_jac[3:6, k] = z_joint

        cl_jac[:, 7] = mat([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
        base_jac[:, 7] = mat([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T

        # compute V
        tess = 8
        V = mat(np.zeros((3,tess)))
        for i in range(tess):
            V[:, i] = mat(np.array([cos(2* pi * (i+1) / tess), sin(2 * pi * (i+1) / tess), 0.0])).T

        #print('R6:', Origins_T[6][0:3, 0:3])
        #print('R7:', Origins_T[7][0:3, 0:3])

        V = Origins_T[6][0:3, 0:3] * V

        return cl_jac, cl_point, V, normalized_shaft_dir, base_jac


    def snake_kin(self, x, BERNS):
        switch = False

        if x > 0:       # the snake kin function only accepts negative x values
            switch = True
            x = -x

        pos = mat([0.0, 0.0, 0.0])

        if BERNS:
            x = np.minimum(np.maximum((x*1000.0 + 18.0)/18.0, 0.0), 1.0)


            pos[0, 0] = 8537.5400685750009870389476418495 * pow(x, 2) * pow(x - 1.0, 5) \
                + 3396.4156994599998142803087830544 * pow(x, 3) * pow(x - 1.0, 4)\
                - 1076.4318034749999242194462567568 * pow(x, 4) * pow(x - 1.0, 3)\
                + 728.29365970200001356715802103281 * pow(x, 5) * pow(x - 1.0, 2)\
                + 13369.137077723000402329489588737 * x * pow(x - 1.0, 6)\
                - 61.503990898000004960977094015107 * pow(x, 6) * (x - 1.0)\
                - 0.78931800299999999026567820692435 * pow(x, 7)\
                + 7546.6795344140000452171079814434 * pow(x - 1.0, 7)

            pos[0, 2] = 73.483180490999998824008798692375 * sin(
            667.48074531064104797940806879664 * pow(x, 2) * pow(x - 1.0, 5)
            + 265.53808992415240541707127864371 * pow(x, 3) * pow(x - 1.0, 4)
            - 84.157438405966357996210301739814 * pow(x, 4) * pow(x - 1.0, 3)
            + 56.939351485122092363701882422881 * pow(x, 5) * pow(x - 1.0, 2)
            + 1045.223976593074672334132763316 * x * pow(x - 1.0, 6)
            - 4.8084962828207292930074425810965 * pow(x, 6) * (x - 1.0)
            - 0.061710348027389578643813541767592 * pow(x, 7)
            + 590.01342773106151642310184964164 * pow(x - 1.0, 7)
            + 1.5468467239999998952271198504604)\
                    - 195.29114571700000624332460574806 * sin(
            140.12779851651771723810170305539 * pow(x, 4) * pow(x - 1.0, 3)
            - 442.13878972716728188061169885953 * pow(x, 3) * pow(x - 1.0, 4)
            - 1111.4003606116601389031760595723 * pow(x, 2) * pow(x - 1.0, 5)
            - 94.807852088838231889168046996875 * pow(x, 5) * pow(x - 1.0, 2)
            - 1740.3682618063638469889608499926 * x * pow(x - 1.0, 6)
            + 8.0064699098393432346403637226158 * pow(x, 6) * (x - 1.0)
            + 0.10275188240702415345090477870919 * pow(x, 7)
            - 982.41206349831999235089623108875 * pow(x - 1.0, 7)
            + 1.6244771259999999379886048700428)\
                    + 156.00152554600001053586311172694 * sin(
            1166.2172758296391470961043868635 * pow(x, 2) * pow(x - 1.0, 5)
            + 463.94612883736452608586662007109 * pow(x, 3) * pow(x - 1.0, 4)
            - 147.03923558563522749786723421713 * pow(x, 4) * pow(x - 1.0, 3)
            + 99.484001363337589851111983187936 * pow(x, 5) * pow(x - 1.0, 2)
            + 1826.2073732882033689827826919661 * x * pow(x - 1.0, 6)
            - 8.4013680921662050279618542508056 * pow(x, 6) * (x - 1.0)
            - 0.10781985019434223382201148868164 * pow(x, 7)
            + 1030.8669684114957295190022294467 * pow(x - 1.0, 7)
            + 1.5125408659999999283485294654383)

            if switch == False:         # To account for correct x (bend direction) in snake base frame, if + snake command
                pos[0, 0] = - pos[0, 0]

            pos[0, 1] = 0

            #print("Snake pos: ", pos/1000.0)


        else:
            # polynomial degree four
            x = x * 1000 # to mm
            pos[0,0] = -0.0048 * pow(x, 4) - 0.1255 * pow(x, 3) - 0.5947 * pow(x,2) + 4.1977 * pow(x,1) - 0.1386
            pos[0,1] = 0.0
            pos[0,2] = 0.0002 * pow(x, 4) - 0.0754 * pow(x, 3) - 1.2515 * pow(x, 2) - 1.5549 * pow(x, 1) + 33.6705

            #if switch == False:         # To account for correct x (bend direction) in snake base frame, if + snake command
            #    pos[0, 0] = - pos[0, 0]


        return pos/1000.0


    def snake_jac(self, x, BERNS):
        switch = False

        if x > 0:   # the jacobian function only accepts negative x
            switch = True
            x = -x

        jac = mat([0.0,0.0,0.0])

        if BERNS:
            x = np.minimum(np.maximum((x*1000.0 + 18.0)/18.0, 0.0), 1.0)

            jac[0, 0] = 52876.947441255004378035664558411 * pow(x, 2) * pow(x - 1.0, 4)\
                    + 9279.9355839399995602434501051903 * pow(x, 3) * pow(x - 1.0, 3)\
                    + 412.1728880850002951774513348937 * pow(x, 4) * pow(x - 1.0, 2)\
                    + 728.29365970200001356715802103281 * pow(x, 5) * (2.0 * x - 2.0)\
                    + 97289.902603488004388054832816124 * x * pow(x - 1.0, 5)\
                    - 369.02394538800002976586256409064 * pow(x, 5) * (x - 1.0)\
                    - 67.029216919000004892836841463577 * pow(x, 6)\
                    + 66195.893818621000718849245458841 * pow(x - 1.0, 6)

            jac[0, 2] = 156.00152554600001053586311172694 * cos(
            1166.2172758296391470961043868635 * pow(x, 2) * pow(x - 1.0, 5)
            + 463.94612883736452608586662007109 * pow(x, 3) * pow(x - 1.0, 4)
            - 147.03923558563522749786723421713 * pow(x, 4) * pow(x - 1.0, 3)
            + 99.484001363337589851111983187936 * pow(x, 5) * pow(x - 1.0, 2)
            + 1826.2073732882033689827826919661 * x * pow(x - 1.0, 6)
            - 8.4013680921662050279618542508056 * pow(x, 6) * (x - 1.0)
            - 0.10781985019434223382201148868164 * pow(x, 7)
            + 1030.8669684114957295190022294467 * pow(x - 1.0, 7)
            + 1.5125408659999999283485294654383) * (
                7222.924765660289313738121794531 * pow(x, 2) * pow(x - 1.0, 4)
                + 1267.6275730069171943519975434158 * pow(x, 3) * pow(x - 1.0, 3)
                + 56.3023000597822667619582132883 * pow(x, 4) * pow(x - 1.0, 2)
                + 99.484001363337589851111983187936 * pow(x, 5) * (2.0 * x - 2.0)
                + 13289.678791388498508088904925524 * x * pow(x - 1.0, 5)
                - 50.408208552997230167771125504833 * pow(x, 5) * (x - 1.0)
                - 9.156107043526600664715934671577 * pow(x, 6)
                + 9042.2761521686734756157982980927 * pow(x - 1.0, 6))\
                    + 195.29114571700000624332460574806 * cos(
            140.12779851651771723810170305539 * pow(x, 4) * pow(x - 1.0, 3)
            - 442.13878972716728188061169885953 * pow(x, 3) * pow(x - 1.0, 4)
            - 1111.4003606116601389031760595723 * pow(x, 2) * pow(x - 1.0, 5)
            - 94.807852088838231889168046996875 * pow(x, 5) * pow(x - 1.0, 2)
            - 1740.3682618063638469889608499926 * x * pow(x - 1.0, 6)
            + 8.0064699098393432346403637226158 * pow(x, 6) * (x - 1.0)
            + 0.10275188240702415345090477870919 * pow(x, 7)
            - 982.41206349831999235089623108875 * pow(x - 1.0, 7)
            + 1.6244771259999999379886048700428) * (
                            6883.4181722398025401577153944403 * pow(x, 2) * pow(x - 1.0, 4)
                            + 1208.0439648425982585700399832166 * pow(x, 3) * pow(x - 1.0, 3)
                            + 53.655864894638007731535125818211 * pow(x, 4) * pow(x - 1.0, 2)
                            + 94.807852088838231889168046996875 * pow(x, 5) * (2.0 * x - 2.0)
                            + 12665.0102920615033597401172191 * x * pow(x - 1.0, 5)
                            - 48.038819459036059407842182335695 * pow(x, 5) * (x - 1.0)
                            - 8.7257330866885123087966971735801 * pow(x, 6)
                            + 8617.2527062946037934452344676139 * pow(x - 1.0, 6))\
                    + 73.483180490999998824008798692375 * cos(
            667.48074531064104797940806879664 * pow(x, 2) * pow(x - 1.0, 5)
            + 265.53808992415240541707127864371 * pow(x, 3) * pow(x - 1.0, 4)
            - 84.157438405966357996210301739814 * pow(x, 4) * pow(x - 1.0, 3)
            + 56.939351485122092363701882422881 * pow(x, 5) * pow(x - 1.0, 2)
            + 1045.223976593074672334132763316 * x * pow(x - 1.0, 6)
            - 4.8084962828207292930074425810965 * pow(x, 6) * (x - 1.0)
            - 0.061710348027389578643813541767592 * pow(x, 7)
            + 590.01342773106151642310184964164 * pow(x - 1.0, 7)
            + 1.5468467239999998952271198504604) * (
                            4134.0179963256624561482541799143 * pow(x, 2) * pow(x - 1.0, 4)
                            + 725.52260607274418968344390761559 * pow(x, 3) * pow(x - 1.0, 3)
                            + 32.224442207711387829878506894963 * pow(x, 4) * pow(x - 1.0, 2)
                            + 56.939351485122092363701882422881 * pow(x, 5) * (2.0 * x - 2.0)
                            + 7606.3053501797301299636127174892 * x * pow(x - 1.0, 5)
                            - 28.850977696924375758044655486579 * pow(x, 5) * (x - 1.0)
                            - 5.2404687190124563435141373734696 * pow(x, 6)
                            + 5175.3179707105052872958457108075 * pow(x - 1.0, 6))

            jac[0, 0] = -jac[0, 0]  # To account for correct x (bend direction) in snake base frame, with + or - snake command

            jac[0,1] = 0.0

            #print("Jac_s: ", jac/1000.0)

        else:
            # 4th order Polynomial
            x = x * 1000
            jac[0,0] = -0.0192 * pow(x, 3) - 0.3765 * pow(x, 2) - 1.1894 * pow(x,1) + 4.1977
            jac[0,1] =  0.0
            jac[0,2] = 0.0008 * pow(x, 3) - 0.2262 * pow(x, 2) - 2.503 * pow(x, 1) - 1.5549


        #print('Snake Jac: ', jac/1000.0)

        return jac/1000.0

    def snake_kin_from_joints(self, th):
        # Assuming length 35 for snake and n joints and n-1 segments of 35/(n-1) length, estimate tip position
        n = 27.0
        l = 34.0 / 1000.0
        sl = l/n

        pos = np.array([0, 0, 0])  # Starting the transformations
        Origin_pos = mat(np.zeros((3,int(n))))

        for i in range(int(n)):
            pos = pos + np.array([sl * sin(np.sum(th[0:i])), 0.0, sl * cos(np.sum(th[0:i]))])
            Origin_pos[:, i] = np.matrix(pos).T

        return Origin_pos

    def objective(self, x):
        n = 27.0
        l = 34.0/1000.0
        sl = l/n

        pos = np.array([0.0, 0.0])

        for i in range(int(n)):
            pos = pos + np.array([sl * math.sin(np.sum(x[0:i])), sl * math.cos(np.sum(x[0:i]))])

        obj = pos - tip
        return obj[0] * obj[0] + obj[1] * obj[1]

    def snake_IK(self, tip_pos):
        global tip
        tip = np.array([tip_pos[0], tip_pos[2]])

        n = 27

        # initialize solution
        x0 = np.zeros([n])

        # Even joint angles
        lb = (-5*math.pi/180.0) * np.ones([n])
        ub = ( 5*math.pi/180.0) * np.ones([n])

        bnds = Bounds(lb, ub)

        solution = minimize(self.objective, x0, method='SLSQP', bounds=bnds)
        #print("Solution: ", solution.x)
        return solution.x
