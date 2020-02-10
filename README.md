Simulation Framework for Constrained Motion Control of Robots
=============================================================

A simulation framework for optimization-based constrained motion control (CMC) of robots (rigid and flexible) is provided. The CMC code is written in Python 3.6 and CoppeliaSim from [Coppelia Robotics](http://www.coppeliarobotics.com/) simulation framework is used for visualization. Task-space motion control of robots with arbitrary constraints in the form of virtual fixtures such as a programmable remote center of motion (RCM) can be achieved by adding the desired constriants to the controller.

Dependencies
------------

This repository  depends on the following libraries: 
* [Numpy](https://numpy.org/)
* [Scipy](https://www.scipy.org/)
* [cvxpy](https://www.cvxpy.org/)

In addition, if you wish to use the visualization capabilities, you need to [install CoppeliaSim](http://www.coppeliarobotics.com/downloads) (previously known as V-REP). Since we use the Python remote API for communication with the simulation environment, after installing CoppeliaSim, you nee to copy the following files from the installation directory (as instructed [here](http://www.coppeliarobotics.com/helpFiles/en/remoteApiClientSide.htm) in the 'Python client' section) to the root directory of this repository.

* sim.py
* simConst.py
* remoteApi.dll, remoteApi.dylib or remoteApi.so (depending on your target platform)

The above files should be found in the CoppeliaSim's installation directory, under _programming/remoteApiBindings/python_.

Virtual Fixtures
----------------
Currently, the following virtual fixtures can be added to the controller in the form of constraints:

* Joint position limits
* Joint velocity limits
* Programmable RCM
* End-effector axis range limit
* Allowable (forbidden) regions defined by hyper-planes

Running Example
---------------

The example provided in this repository ('cmc\_example.py') instructs you on how you can initialize the robotic system configuration, add the desired constriants and the set of desired target points for motion control. After installing all the dependencies, you can run the example code by:

1. Start CoppeliaSim by running coppeliaSim.sh in the installation directory.
2. In the simulation, under the 'File' tab, click on 'Open Scene ...'.
3. Load 'simulation\_example.ttt' scene provided in this repository under the 'simulation' folder.
4. Hit play button in simulation.
5. Now run: python cmc\_example.py

If everything is successful, the robot in the simulation starts moving to the specified target points while satisfying all constraints (joing position and velocity limits, RCM, end-effector axis range limit, and two hyper-plane virtual fixtures).
