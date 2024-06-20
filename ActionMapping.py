'''
Interpolation for action mapping

@Author: Yuanda Wang

@Date: May 10, 2022

@Modi: 
- May 19, 2022: add value clip spd and steer

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
import time

class ActionMappingClass():
    def __init__(self):
        self.spd_num   = 200
        self.steer_num = 200
        self.dir_num = 200
        self.amp_num = 200

        self.spd_max = 30
        self.steer_max = 35.0 / 180.0 * np.pi
        self.dir_max = np.pi * 2
        self.amp_max = np.sqrt(2)

        self.actionmap = np.load('actionmap_200.npy')

    # action mapping
    def mapping(self, spd, steer, ux, uy):
        # clip values before send to interpolation
        if spd < 0: spd = 0
        if spd > self.spd_max: spd = self.spd_max
        if steer > self.steer_max: steer = self.steer_max
        if steer < -self.steer_max: steer = -self.steer_max
        # process with interpolate and mapping
        self.amp = amp = self.interpolate(spd, steer, ux, uy)
        ux_map = amp * np.cos(self.theta)
        uy_map = amp * np.sin(self.theta)
        if abs(ux_map) > abs(ux) or abs(uy_map) > abs(uy):
            return [ux, uy]
        else:
            return [ux_map, uy_map]


    # inverse function for i_spd, i_steer, i_dir, then find amp
    def interpolate(self, spd, steer, ux ,uy):
        # invert spd = i_spd / spd_num * spd_max
        x_spd = spd / self.spd_max * (self.spd_num-1)  
        i_spd_0 = int(np.around(x_spd))
        i_spd_1 = i_spd_0 - 1
        if i_spd_1 < 0 : i_spd_1 = i_spd_0 + 2
        i_spd_2 = i_spd_0 + 1
        if i_spd_2 > self.spd_num-1 : i_spd_2 = i_spd_0 - 2

        # invert steer = ((i_steer/(steer_num-1))-0.5)*2*steer_max
        x_steer = (steer / (2*self.steer_max) + 0.5) * (self.steer_num-1)
        i_steer_0 = int(np.around(x_steer))
        i_steer_1 = i_steer_0 - 1
        if i_steer_1 < 0: i_steer_1 = i_steer_0 + 2
        i_steer_2 = i_steer_0 + 1
        if i_steer_2 > self.steer_num-1 : i_steer_2 = i_steer_0 - 2

        # get angle 
        self.theta = theta = np.arctan2(uy, ux)
        if theta < 0 : theta += 2 * np.pi
        # invert dir
        x_dir = (theta / self.dir_max) * (self.dir_num-1)
        i_dir_0 = int(np.around(x_dir))
        i_dir_1 = i_dir_0 - 1
        if i_dir_1 < 0: 
            i_dir_1 = self.dir_num-1
            i_dir_1x = i_dir_0 - 1
        else:
            i_dir_1x = i_dir_1
        i_dir_2 = i_dir_0 + 1
        if i_dir_2 > self.dir_num-1: 
            i_dir_2 = 0
            i_dir_2x = i_dir_0 + 1 
        else:
            i_dir_2x = i_dir_2

        # print('spd:', spd, 'i_spd:', i_spd_1, i_spd_0, i_spd_2)
        # print('steer:', steer, 'i_steer', i_steer_1, i_steer_0, i_steer_2)
        # print('dir:', theta, 'i_dir:', i_dir_1, i_dir_0, i_dir_2)

        spd_list   = np.sort([i_spd_0, i_spd_1, i_spd_2])
        steer_list = np.sort([i_steer_0, i_steer_1, i_steer_2])
        # dir_list   = [i_dir_0, i_dir_1, i_dir_2]  # no overlimit index 
        dirx_list  = np.sort([i_dir_0, i_dir_1x, i_dir_2x])  # have overlimit index

        # get amp values 
        values = np.zeros([3,3,3])
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    i_spd = spd_list[x]
                    i_steer = steer_list[y]
                    i_dir = dirx_list[z]
                    # handle overlimit i_dir values
                    if i_dir < 0: i_dir = self.dir_num-1
                    if i_dir > self.dir_num-1: i_dir = 0
                    values[x, y, z] = self.actionmap[i_spd, i_steer, i_dir]

        # interpolation with scipy: interpn(points, values, point)
        points = (spd_list, steer_list, dirx_list)
        # values = find_amp_value(*np.meshgrid(*points, indexing='ij'))
        point = (x_spd, x_steer, x_dir)
        inter_amp = interpn(points, values, point)
        return inter_amp[0]


if __name__ == '__main__':

    amap = ActionMappingClass()
    i_spd = 0
    i_steer = 0
    spd = i_spd / (amap.spd_num-1) * amap.spd_max + 0 # test
    steer = ((i_steer/(amap.steer_num-1))-0.5)*2*amap.steer_max + 0 # test

    # spd = 0
    # steer = 0
    # steer = 10.0 / 180.0 * np.pi
    # theta = 360 / 180 * np.pi
    # ux = np.cos(theta)
    # uy = np.sin(theta)
    ux = -1
    uy = -1
    t1 = time.time()
    amp = amap.interpolate(spd, steer, ux, uy)
    t2 = time.time()
    print('spd: %.1f' % spd)
    print('steer: %.1f' % (steer/np.pi*180))
    print('amp: %.1f' % amp)
    print('inter time:', t2-t1)

    # draw test
    fig2 = plt.figure()
    spd = i_spd / (amap.spd_num-1) * amap.spd_max
    steer = ((i_steer/(amap.steer_num-1))-0.5)*2*amap.steer_max
    amplist = amap.actionmap[i_spd, i_steer]
    # for i in range(len(amplist)):
    #     theta = (i/len(amplist)) * 2 * np.pi
    #     x = amplist[i] * np.cos(theta)
    #     x = np.clip(x, -1, 1)
    #     y = amplist[i] * np.sin(theta)
    #     y = np.clip(y, -1, 1)
    #     plt.plot([0, x], [0, y], 'r-')
    xlist, ylist = [], []
    for i in range(amap.amp_num):
        theta = (i/(amap.dir_num-1)) * amap.dir_max
        x = amplist[i] * np.cos(theta)
        y = amplist[i] * np.sin(theta)
        # plt.plot([0, x], [0, y], 'r-')
        xlist.append(x)
        ylist.append(y)
    plt.plot(xlist, ylist, 'r-')

    # draw interpolated line
    theta2 = np.arctan2(uy, ux)
    if theta2 < 0: theta2 += 2 * np.pi
    plt.plot([0, amp*np.cos(theta2)],[0, amp*np.sin(theta2)], 'g-')

    plt.axis('equal')
    plt.show()


# def value_func_3d(x, y, z):
#     return 2 * x + 3 * y - z

# x = np.linspace(0, 4, 5)
# y = np.linspace(0, 5, 6)
# z = np.linspace(0, 6, 7)

# points = (x, y, z)
# values = value_func_3d(*np.meshgrid(*points, indexing='ij'))

# point = np.array([2.21, 3.12, 1.15])
# print(interpn(points, values, point))
# print(value_func_3d(2.21, 3.12, 1.15))