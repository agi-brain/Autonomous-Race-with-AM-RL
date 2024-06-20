"""

SimpleTrack Car Race Env 

@Author: Yuanda Wang

@Date: Mar. 30, 2024

"""

import numpy as np
import random
import matplotlib.pyplot as plt
from CarModel_Kinematic import CarModelClass
from SimpleTrack import SimpleTrackClass

class SimpleTrackEnvClass():
    def __init__(self):
        self.track = SimpleTrackClass()

    def reset(self):
        # reset car at random state
        pose = self.track.random_car_pose()
        spd = np.random.uniform(10, 20)
        self.car = CarModelClass(pose, spd)
        self.reset_flags()
        ob = self.observe()
        return ob

    def test_reset(self):
        # reset car at start point
        posx, posy = 0, 0
        psi = np.pi / 2
        spd = 10
        self.car = CarModelClass([posx, posy, psi], spd)
        self.reset_flags()
        ob = self.observe()
        return ob

    def step(self, action):
        # move the car by action
        self.car.step(action)

        # observe car and track states
        ob = self.observe()

        # check fails
        self.check_fails()
        done = self.FAIL

        # give reward
        r = self.reward()

        return ob, r, done

    def observe(self):
        # observe car state
        pose = self.car.pose
        self.spd = spd = self.car.spd
        psi_dot = self.car.psi_dot
        steer = self.car.steer

        # observe car-in-track states
        if self.track.findcar(pose):
            # off-centerline distance
            self.dist = dist = self.track.centerlinedist
            # car-track angle
            self.angle00 = angle00 = self.track.find_cartrack_angle(pose)
            # looking forward track edge
            centerpoints = []
            # point of 5m
            addtrip = 5
            # print('addtrip:', addtrip)
            pt = self.track.find_relative_centerpoint(pose, addtrip)
            ptlist = [pt[0]/100, pt[1]/100]
            centerpoints += ptlist
            # points of [10, 20, 30, 40]
            for i in range(1,5):
                addtrip = i*10
                # print('addtrip', addtrip)
                pt = self.track.find_relative_centerpoint(pose, addtrip)
                ptlist = [pt[0]/100, pt[1]/100]
                centerpoints += ptlist
            for i in range(3,11):
                addtrip = i*20
                # print('addtrip', addtrip)
                pt = self.track.find_relative_centerpoint(pose, addtrip)
                ptlist = [pt[0]/100, pt[1]/100]
                centerpoints += ptlist
        else:
            raise RuntimeError('Car out of track, unable to get states!')

        # normalize observations
        spd /= 30.0         # 30m/s is about the max speed
        dist /= self.track.width/2
        psi_dot /= 1.57     # np.pi/2 rad/s is max rotation speed
        steer /= np.pi/4    # np.pi/4 rad is max steer angle
        angle00 /= np.pi

        ob = [spd, psi_dot, steer, dist, angle00] + centerpoints
              
        return ob

    def reward(self):
        if not self.FAIL:
            r = self.spd * np.cos(self.angle00) / 10.0
        else:
            r = -100
        return r

    def reset_flags(self):
        self.OUT_TRACK = False
        self.WRONG_DIR = False
        self.MAX_ACC = False
        self.STOP = False
        self.FAIL = False

    def check_fails(self):
        # check wrong way direction
        if abs(self.angle00) > np.pi / 2:
            self.WRONG_DIR = True
            self.FAIL = True
        # check out of track 
        if abs(self.dist) > self.track.width / 2 * 0.95:
            self.OUT_TRACK = True
            self.FAIL = True
        # check max acc
        if self.car.check_acc():
            self.MAX_ACC = True
            self.FAIL = True

        if self.car.spd < 6.0:  # below 6 is too slow, for most of the time, should start from 10
            self.STOP = True
            self.FAIL = True

    def query_fail_reason(self):
        if self.OUT_TRACK: 
            reason = 'OUT TRACK!'
        elif self.MAX_ACC:   
            reason = 'MAX_ACC'
        elif self.WRONG_DIR: 
            reason = 'WRONG DIRECTION'
        elif self.STOP:      
            reason = 'STOP'
        else:
            reason = 'FINISH'
        return reason

if __name__ == '__main__':
    env = SimpleTrackEnvClass()
    ob = env.reset()
    for step in range(10):
        # generate action
        action = [0,0]
        ob, r, done = env.step(action)
        print('ob', ob)
        print('obsize', np.shape(ob))
        print('r', r)


