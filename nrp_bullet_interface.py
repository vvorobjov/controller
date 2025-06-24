"""
Derived from bullet_simulator in examples for pysim engine with bullet at:
https://bitbucket.org/hbpneurorobotics/nrp-core/src/master/examples/pysim_examples/bullet_control/

A Py_Sim Engine for a PyBullet car simulation
--> obtain information from simulation and send them to controller engine
--> receive controller command from engine to run the simulation
"""
from nrp_core.engines.py_sim import PySimEngineScript

import arm_1dof.robot_arm_1dof
#import arm_1dof.robot_arm_muscle_1dof
import pybullet
import os
import numpy as np
from nrp_core.engines.py_sim.SimManager import SimulatorManager


class Script(PySimEngineScript):

    def __init__(self):
        super().__init__()



    def initialize(self):
        print("PyBullet Engine Server is initializing.")
        # Initialize datapack of sensors and controller
        self._registerDataPack("positions")
        
        self._registerDataPack("control_cmd")
        print("DataPacks registered.")

        print('self.sim_manager.get_model_properties("Body")', self.sim_manager.get_model_properties("Body"))
        print('self.sim_manager.get_model_property(0, "Body")', self.sim_manager.get_model_property(0, "Body"))
        print('self.sim_manager.get_model_property(1, "Body")', self.sim_manager.get_model_property(1, "Body"))
        print('self.sim_manager.get_model_properties("Link")', self.sim_manager.get_model_properties("Link"))

        bodies = self.sim_manager.get_model_properties("Body")

        self._skel_arm = arm_1dof.robot_arm_1dof.RobotArm1Dof(
            bullet_ctrl=self.sim_manager.sim_interface.physics_client,
            bullet_body_id=bodies['skeleton']
        )

        pybullet.setGravity(0, 0, 0)

        '''
        
        self.joint_num = pybullet.getNumJoints(self.model,
                                               physicsClientId=self.phys_client)
        self.body_num = pybullet.getNumBodies(physicsClientId=self.phys_client)
        self.joint_name_to_id = {}
        self.link_name_to_id = {}
        self.body_name_to_id = {}
        for i in range(self.joint_num):
            joint_info = pybullet.getJointInfo(self.model, i)

            self.joint_name_to_id[joint_info[1].decode('UTF-8')] = i  # JOINT_NAME = 1
            self.link_name_to_id[joint_info[12].decode('UTF-8')] = i  # LINK_NAME = 12

        for i in range(self.body_num):
            body_info = pybullet.getBodyInfo(i)
            self.body_name_to_id[body_info[1].decode('UTF-8')] = i  # JOINT_NAME = 1

        print("Joints: ", self.joint_name_to_id)
        print(" \n \n \n ")
        print("Links: ", self.joint_name_to_id)
        print(" \n \n \n ")
        '''

        
        #self.sim_time = 0  # seconds

        self.upperarm = pybullet.getLinkState(self._skel_arm._body_id, self._skel_arm.UPPER_ARM_LINK_ID,
                                      computeLinkVelocity=True)[0]
        self.forearm = pybullet.getLinkState(self._skel_arm._body_id, self._skel_arm.FOREARM_LINK_ID,
                                      computeLinkVelocity=True)[0]
        self.hand = pybullet.getLinkState(self._skel_arm._body_id, self._skel_arm.HAND_LINK_ID,
                                      computeLinkVelocity=True)[0]
        # hand = bullet_robot.EEPose()[0]
        self._init_pos = [self.hand[0], self.hand[2]]
        self.rho = np.linalg.norm(np.array(self.upperarm)-np.array(self.hand))
        self.z = self.upperarm[2] - self.rho * np.cos(1.57)
        self.y = self.upperarm[0] + self.rho * np.sin(1.57)
        self._tgt_pos  = [self.y,self.z]

        self._setDataPack("positions", {"hand": self.hand, "upperarm": self.upperarm}) # Bullet: “Body”, “Joint”, “Link”



    def runLoop(self, timestep):
        print("timestep = ", timestep)
        #  Receive control data
        self.action = self._getDataPack("control_cmd").get("act_list")

        print("Action received control_cmd['act_list']: \n", self.action,"\n \n")
        
        act_list = {
            "Joint": [
                {
                    'index': self._skel_arm.UPPER_ARM_LINK_ID,
                    'controlMode': 'VELOCITY_CONTROL',
                    'force': 1,
                    'targetVelocity': 5,
                    'velocityGain': self.action[0]
                }
            ]
        }
        

        # Execute
        '''    
        if self._skel_muscles:
            # Update muscle joint states
            self._skel_arm.UpdateStats()
            self._skel_muscles.SetJointPos(JOINT_INDEX(0), self._skel_arm.JointPos(0))
            self._skel_muscles.SetJointPos(JOINT_INDEX(1), self._skel_arm.JointPos(1))
            self._skel_muscles.SetJointVel(JOINT_INDEX(0), self._skel_arm.JointVel(0))
            self._skel_muscles.SetJointVel(JOINT_INDEX(1), self._skel_arm.JointVel(1))

            self._skel_muscles.Integrate(self._timestep)

            # Apply muscle torques to robot
            torques = [ self._skel_muscles.GetTorque(JOINT_INDEX(0)), self._skel_muscles.GetTorque(JOINT_INDEX(1))]
            self._skel_arm.SetJointTorques(joint_ids=[0,1], torques=torques)
        '''


        self.sim_manager.run_step(act_list, timestep) # WARNING: the timestep is given in seconds bcause BulletLib has been modified

        # Collect observation information
        self._skel_arm.UpdateStats()
        self.hand = pybullet.getLinkState(self._skel_arm._body_id, self._skel_arm.HAND_LINK_ID,
                                      computeLinkVelocity=True)[0]
        
        self.upperarm = pybullet.getLinkState(self._skel_arm._body_id, self._skel_arm.UPPER_ARM_LINK_ID,
                                      computeLinkVelocity=True)[0]

        self._setDataPack("positions", { "hand": self.hand, "upperarm": self.upperarm } )
        
    

    
    def shutdown(self):
        self.sim_manager.shutdown()
        print("Simulation End !!!")
