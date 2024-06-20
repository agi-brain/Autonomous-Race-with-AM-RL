"""

New Bicycle Model 
-- Longitudinal Dynamic
-- Lateral Kinematic

@Author: Yuanda Wang

@Date: May 9, 2022

# Action
a = [force, steer_dot]
   - force: driven force, acc or brake
   - steer_dot: angular velocity of turn the steering wheel 

"""

import numpy as np


# ---------------------- Car Parameters ------------------# 
GravityAcc = 9.81
# Shape and Mass
CarWheelBase = 2.94    # axle length
CarLenF = 1.17
CarLenR = 1.77
CarMass = 1860.0  # mass of vehicle 1500 kg

# Tire 
TireStiff = 54600 # N/rad
TireRadius = 0.4572 # 18-inch


# Drag / friction 
CarAirResist = 0.3
AirDense = 1.2258
CarFrontAera = 2.05
TireRotateFriction = 0.015

# Steering
# max steering angle vary from 30-40 deg, inside wheel and outside wheel are different
MaxSteer = 35.0 / 180.0 * np.pi 
MaxSteerRate = MaxSteer  # from -MaxSteer to MaxSteer in 2 second

# Driven and Brake Force
# fixed value
MotorPowerMax = 125000.0   # 125kw 
MotorTorqueMax = 310    # Nm
K_drive = 10  # main gear ratio
# calculated value:
ForceMax = K_drive * MotorTorqueMax / TireRadius  
MotorBaseSpd = MotorPowerMax / ForceMax  # V = P / F 

K_brake = 0.9 * GravityAcc  # maximum brake decelration. Generally: 0.85G - 1.0G
AccMax  = 1.2 * GravityAcc  # 1.1 G for construct the action map, 1.2 G for training

#---- Other parameters ------#


# RK4 function for simulation
def RK4(ufunc, x0, u, h):
    k1 = ufunc(x0, u)
    k2 = ufunc(x0 + h*k1/2, u)
    k3 = ufunc(x0 + h*k2/2, u)
    k4 = ufunc(x0 + h*k3, u)
    x1 = x0 + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return x1

class CarModelClass():

    def __init__(self, pose0, spd0):
        self.pose = pose0  # [x, y, phi]
        self.spd = spd0
        self.dt = 0.01
        self.steer = 0.0
        self.psi_dot = 0.0
        self.ref_dist = 0.0
        self.ref_spd = 0.0
        self.temp_trip = 0.0  # for oppo, to find the nearest front oppo 
        self.temp_angle = 0.0 # for debug, find the control bug

    def reset(self, pose0, spd0):
        self.pose = pose0
        self.spd = spd0
        self.steer = 0.0
        self.psi_dot = 0.0

    def AM_reset(self, spd, steer):
        self.pose = [0.0, 0.0, 0.0]
        self.psi_dot = 0.0
        self.spd = spd
        self.steer = steer

    def convert_control(self, action):
        # convert power/brake to force
        ux = action[0]  # [-1, 1]
        if ux > 0: 
            # motor -- accelerate
            # under base speed
            if self.spd < MotorBaseSpd:   # torque is prop to ux
                torque = K_drive * MotorTorqueMax * ux
            else: # torque is limited by max power
                torque_req = K_drive * MotorTorqueMax * ux
                # print('torque_req:', torque_req)
                torque_max = TireRadius * MotorPowerMax / self.spd
                # print('torque_max:', torque_max)
                torque = torque_req if torque_req < torque_max else torque_max
                # print('torque_out:', torque)
            self.force = torque / TireRadius
            # print('force:', self.force)
        else:
            # brake -- decelerate
            brake = ux
            self.force = brake * (K_brake * CarMass)

        # convert steer_rate to steer angle
        steer_rate = action[1] * MaxSteerRate # [-1, 1]
        self.steer = self.crop_steer(self.steer + steer_rate * self.dt)    

    # dynamic functions
    def longitudinal_dynamic(self):
        # get drags
        air_drag = self.get_air_drag()
        tire_drag = self.get_rotation_drag()
        # get acc
        self.long_force = self.force - air_drag - tire_drag
        self.long_acc = self.long_force / CarMass
        # update speed
        self.spd += self.long_acc * self.dt
        if self.spd < 0: 
            self.spd = 0
        if self.spd > 30:
            self.spd = 30

    def lateral_kinematic(self):
        if abs(self.steer) > 0.0001:
            # get slip angle
            self.beta = np.arctan((CarLenR*np.tan(self.steer))/CarWheelBase)
            # get turn radius
            self.radius = CarWheelBase / (np.tan(self.steer) * np.cos(self.beta))
            # get lat acc
            self.lat_force = (CarMass * self.spd * self.spd) / self.radius
            self.lat_acc = self.lat_force / CarMass
            # get yaw rate 
            self.psi_dot = self.spd / self.radius
        else:
            self.beta = 0
            self.radius = 1e10
            self.lat_force = 0
            self.lat_acc = 0
            self.psi_dot = 0
        # update yaw angle
        self.psi = self.pose[2] + self.psi_dot * self.dt

    def update_pose(self):
        x, y, psi = self.pose[0], self.pose[1], self.pose[2]
        x_dot = self.spd * np.cos(self.psi + self.beta)
        x += x_dot * self.dt
        y_dot = self.spd * np.sin(self.psi + self.beta)
        y += y_dot * self.dt
        # update pose
        self.pose = [x, y, self.psi]

    def step(self, action):
        # convert driver action to ctrl force and steer
        self.ctrl = self.convert_control(action)  # --> update force, steer angle
        # longitudinal direction
        self.longitudinal_dynamic()
        # lateral direction
        self.lateral_kinematic()
        # pose update
        self.update_pose()
        # check sum acc constraint -- move to env
        # FAIL_ACC = self.check_sum_acc()
        # return FAIL_ACC

    def get_air_drag(self):
        v = self.spd
        return 0.5 * CarAirResist * CarFrontAera * AirDense * v * v

    def get_rotation_drag(self):
        if self.spd > -0.01:
            return CarMass * GravityAcc * TireRotateFriction
        else:
            return 0
    
    def crop_steer(self, steer):
        if steer >  MaxSteer: steer =  MaxSteer
        if steer < -MaxSteer: steer = -MaxSteer
        return steer

    def check_acc(self):
        self.acc_sum = np.sqrt(self.lat_acc**2 + self.long_acc**2)
        if self.acc_sum > AccMax:
            return True
        else:
            return False