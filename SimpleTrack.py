"""
Update From: FullTrack.py

Simple Track

@ Yuanda Wang
@ Nov. 11. 2023

Simple Track for single or multi-car race

"""
import numpy as np
import random
import matplotlib.pyplot as plt

def torad(deg):
    return deg / 180 * np.pi

def todeg(rad):
    return rad / np.pi * 180

def get_distance(pt1, pt2):
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def get_vector_length(v):
    x, y = v[0], v[1]
    return np.sqrt(x*x + y*y)

def get_bearing(pt0, pt1):
    x0, y0 = pt0[0], pt0[1]
    x1, y1 = pt1[0], pt1[1]
    x = x1 - x0
    y = y1 - y0
    return np.arctan2(y, x)

''' -edit: Dec. 5, 2023. add a while loop  '''
def adjust_angle(angle):
    while angle > np.pi or angle < -np.pi:   # keep adjust angle until it falls in (-pi, pi]
        if angle > np.pi:
            angle -= 2 * np.pi
        if angle < -np.pi:
            angle += 2 * np.pi
    return angle

def get_vector_angle(v1, v2):
    x1, y1 = v1[0], v1[1]
    x2, y2 = v2[0], v2[1]
    v1_len = get_vector_length(v1)
    v2_len = get_vector_length(v2)
    cos_angle = (x1*x2 + y1*y2)/(v1_len*v2_len)
    angle = np.arccos(cos_angle)
    return angle

def get_angle_diff(a1, a2):
    diff = a1 - a2
    return adjust_angle(diff)

class TrackUnitClass():
    def __init__(self, width, len1, curve_angle, radius, \
                 zero_pos, zero_angle, rotate_angle, start_trip):
        self.width = width
        self.len1 = len1
        self.angle = curve_angle
        self.radius = radius
        self.zero_pos = zero_pos
        self.zero_angle = zero_angle
        self.rotate_angle = rotate_angle
        self.start_trip = start_trip
        self.generate_keypoints()

    def rotate(self, pt):
        a = self.rotate_angle
        x0, y0 = self.zero_pos[0], self.zero_pos[1]
        x1, y1 = pt[0], pt[1]
        x2 = (x1-x0) * np.cos(a) - (y1-y0) * np.sin(a) + x0
        y2 = (x1-x0) * np.sin(a) + (y1-y0) * np.cos(a) + y0
        return [x2, y2]


    def generate_keypoints(self):
        z = self.zero_pos
        w = self.width
        # straight_in 
        # start from np.pi/2 rad
        self.inlineL1 = self.rotate([z[0]-w/2, z[1]])
        self.inlineL2 = self.rotate([z[0]-w/2, z[1]+self.len1])
        self.inlineC1 = self.rotate([z[0],   z[1]])
        self.inlineC2 = self.rotate([z[0],   z[1]+self.len1])
        self.inlineR1 = self.rotate([z[0]+w/2, z[1]])
        self.inlineR2 = self.rotate([z[0]+w/2, z[1]+self.len1])

        # curve center--curve out point direction 
        if self.angle < 0: # right turn
            self.curve_center = self.rotate([self.radius+z[0], self.len1+z[1]])
            self.endpoint_angle = np.pi/2 - (- self.angle - self.zero_angle)   # convert to curve center angle
            self.startpoint_angle = self.endpoint_angle - self.angle
        else: # left turn
            self.curve_center = self.rotate([-self.radius+z[0], self.len1+z[1]])
            self.endpoint_angle = self.angle + self.zero_angle - np.pi/2   # convert to curver center angle
            self.startpoint_angle = self.endpoint_angle - self.angle

        # radius of track left/center/right lines
        self.curve_L = self.radius + self.width / 2 
        self.curve_C = self.radius
        self.curve_R = self.radius - self.width / 2
        
        Lx = self.curve_L * np.cos(self.endpoint_angle) + self.curve_center[0]
        Ly = self.curve_L * np.sin(self.endpoint_angle) + self.curve_center[1]
        self.curve_out_L = [Lx, Ly]
        Cx = self.curve_C * np.cos(self.endpoint_angle) + self.curve_center[0]
        Cy = self.curve_C * np.sin(self.endpoint_angle) + self.curve_center[1]
        self.curve_out_C = [Cx, Cy]
        Rx = self.curve_R * np.cos(self.endpoint_angle) + self.curve_center[0]
        Ry = self.curve_R * np.sin(self.endpoint_angle) + self.curve_center[1]
        self.curve_out_R = [Rx, Ry]

        # curve out direction angle in map frame
        self.out_angle = self.zero_angle +self.angle

        # curve center--curve out point direction (for drawing and positioning)
        if self.angle < 0: # right turn
            self.angle_startpoint = self.out_angle + np.pi/2
            self.angle_endpoint = self.out_angle + np.pi/2 - self.angle
        else:
            self.angle_startpoint = self.zero_angle - np.pi/2
            self.angle_endpoint = self.zero_angle - np.pi/2 + self.angle
        # if angle too small, convert to [-pi, pi]
        # if self.angle_startpoint < -np.pi:
        #     self.angle_startpoint += 2*np.pi
        #     self.angle_endpoint   += 2*np.pi
            
        # print('angle_endpoint:', self.angle_endpoint)
        # print('enpoint_angle:', self.endpoint_angle)
        
        # length and trip
        self.unit_length = self.len1 + self.radius * abs(self.angle)
        self.end_trip = self.start_trip + self.unit_length

    def connect_info(self):
        out_pos = self.curve_out_C
        out_angle = self.out_angle
        rotate_angle = self.out_angle - np.pi/2 # relative to going forward dir (np.pi)
        end_trip = self.end_trip
        return out_pos, out_angle, rotate_angle, end_trip

    def findcar(self, pos):
        self.centerlinepoint = []
        self.centerlineside = 0  # left: +1 ; right -1; out: 0
        self.centerlinepart = 0  # straight: 1 ; curve: 2
        if self.findcar_in_straight(pos):
            self.centerlinepart = 1
            # print('car in straight!')
        elif self.findcar_in_curve(pos):
            self.centerlinepart = 2
            # print('car in curve')
        else:
            return False
        # get track direction
        self.track_dir = self.get_track_direction()
        return True

    def findcar_in_straight(self, pos):
        # move car to unit1 (0,0,pi/2) unit then judge easily
        x, y = pos[0], pos[1]
        movex, movey = self.zero_pos[0], self.zero_pos[1]
        # print('rotate angle:', self.rotate_angle)
        a = -self.rotate_angle
        # move car position
        x1 = x - movex
        y1 = y - movey
        # rotate car position
        x2 = x1 * np.cos(a) - y1 * np.sin(a)
        y2 = x1 * np.sin(a) + y1 * np.cos(a)
        # judge if car inside straight track
        if (x2 > -self.width/2) and (x2 < self.width/2) and (y2 >= 0) and (y2 <= self.len1):
            # inside! get centerlinepoint then move and rotate back
            # centerline position
            cx0 = 0
            cy0 = y2
            # rotate centerlinepoint 
            a = self.rotate_angle
            cx = cx0 * np.cos(a) - cy0 * np.sin(a)
            cy = cx0 * np.sin(a) + cy0 * np.cos(a)
            # move centerlinepoint
            cx = cx + movex
            cy = cy + movey
            self.centerlinepoint = [cx, cy]
            # judge centerlineside
            self.centerlineside = 1 if x2 < 0 else -1
            # get to-centerline dist 
            self.centerlinedist = get_distance(pos, self.centerlinepoint) * self.centerlineside
            # get unit trip 
            self.unit_trip = get_distance(self.zero_pos, self.centerlinepoint)
            return True 
        else:
            return False

    def findcar_in_curve(self, pos):
        # judge from to center distance
        dist = get_distance(pos, self.curve_center)
        if (dist > self.radius + self.width/2) or (dist < self.radius - self.width/2):
            return False
        # judge from bearing angle
        bearing = get_bearing(self.curve_center, pos)
        # print('bearing:', bearing / np.pi * 180)
        # print('start angle:', self.angle_startpoint / np.pi * 180)
        # print('end_angle:', self.angle_endpoint / np.pi * 180)
        if ((bearing > self.angle_startpoint) and (bearing < self.angle_endpoint)) \
           or ((bearing > self.angle_startpoint+2*np.pi) and (bearing < self.angle_endpoint+2*np.pi)) \
           or ((bearing > self.angle_startpoint-2*np.pi) and (bearing < self.angle_endpoint-2*np.pi)):
            # in curve, find centerlinepoint
            ptx = self.curve_center[0] + (self.radius) * np.cos(bearing)
            pty = self.curve_center[1] + (self.radius) * np.sin(bearing)
            self.centerlinepoint = [ptx, pty]
            # judge pos in centerline side
            if self.angle < 0:
                if dist < self.radius:
                    self.centerlineside = -1
                else:
                    self.centerlineside = 1
            else:
                if dist < self.radius:
                    self.centerlineside = 1
                else:
                    self.centerlineside = -1
            # get to-centerline dist 
            self.centerlinedist = get_distance(pos, self.centerlinepoint) * self.centerlineside
            # get unit trip 
            self.unit_trip = self.len1 + self.get_curve_center_angle() * self.radius
            return True
        else:
            return False

    def get_curve_center_angle(self, setpoint=False, trackpoint=[]):
        ''' get the angle between curve_start_point -- curve_center to 
            centerlinepoint -- curve_center
            used to calculate in curve trip and forward  direction
        '''
        if setpoint:
            clp = trackpoint
        else:
            clp = self.centerlinepoint
        zp  = self.inlineC2  # curve start point
        ccp = self.curve_center
        # vector of start point -- curve_center
        v1 = [zp[0]-ccp[0], zp[1]-ccp[1]]
        # vector of centerlinepoint -- curve_center
        v2 = [clp[0]-ccp[0], clp[1]-ccp[1]]
        angle = get_vector_angle(v1, v2)
        # should distingush 0 and np.pi, might be problem when testing...
        return angle
    
    def get_track_direction(self):
        ''' give a postion in curve, output the track direction '''
        if self.centerlinepart == 1:  # in straight line
            track_dir = self.zero_angle
        elif self.centerlinepart == 2: # in curve
            track_dir = self.get_in_curve_direction()
        else:
            raise RuntimeError('car not in track, cannot get track direction')
        return track_dir

    def get_in_curve_direction(self):
        ''' get the track direction in curve '''
        a = self.get_curve_center_angle()
        if self.angle < 0: # right turn
            track_dir = self.zero_angle - a 
        else: # left 
            track_dir = self.zero_angle + a
        return track_dir

    def get_point_track_direction(self, pt, trip):
        ''' get track direction of a point ''' 
        # find the point in curve or straight line
        straight_trip = self.len1
        curve_trip = abs(self.angle) * self.radius
        if trip <= straight_trip: # point in straight line
            track_dir = self.zero_angle
        elif trip <= straight_trip + curve_trip:  # point in curve
            a = self.get_curve_center_angle(setpoint=True, trackpoint=pt)
            if self.angle < 0: # right turn
                track_dir = self.zero_angle - a
            else:  # left turn
                track_dir = self.zero_angle + a
        else:
            raise RuntimeError('forward trip point not in this unit')
        return track_dir

    def trip_to_centerlinepoint(self, trip):
        ''' input in-unit trip, output centerlinepoint'''
        straight_trip = self.len1
        curve_trip = abs(self.angle) * self.radius
        if trip <= straight_trip: # point in straight
            cx, cy = self.zero_pos[0], self.zero_pos[1]
            a = self.zero_angle
            pt = [cx + trip * np.cos(a), cy + trip * np.sin(a)]
        elif trip <= straight_trip + curve_trip: # point in curve
            covered_length = trip - straight_trip
            cover_angle = covered_length / self.radius
            # print('cover_angle:', todeg(cover_angle))
            # print('start_angle:', todeg(self.startpoint_angle))
            if self.angle < 0: # right turn 
                a = self.startpoint_angle - cover_angle
            else: # left turn
                a = self.startpoint_angle + cover_angle
            cx, cy = self.curve_center[0], self.curve_center[1]
            pt = [cx + self.radius * np.cos(a), cy + self.radius * np.sin(a)]
        else: # point not in this track unit
            raise RuntimeError('forward trip point not in this unit!')
        return pt

    def trip_to_edgelinepoint(self, trip):
        ''' input  in-unit trip, output a pair of edge-line-point'''
        straight_trip = self.len1
        curve_trip = abs(self.angle) * self.radius
        if trip <= straight_trip: # point in straight
            xL, yL = self.inlineL1[0], self.inlineL1[1]
            xR, yR = self.inlineR1[0], self.inlineR1[1]
            a = self.zero_angle
            ptL = [xL + trip * np.cos(a), yL + trip * np.sin(a)]
            ptR = [xR + trip * np.cos(a), yR + trip * np.sin(a)]
        elif trip <= straight_trip + curve_trip: # point in curve
            covered_length = trip - straight_trip
            cover_angle = covered_length / self.radius
            cx, cy = self.curve_center[0], self.curve_center[1]
            w = self.width / 2
            if self.angle < 0: # right turn 
                a = self.startpoint_angle - cover_angle
                ptL = [cx + (self.radius + w) * np.cos(a), cy + (self.radius + w) * np.sin(a)]
                ptR = [cx + (self.radius - w) * np.cos(a), cy + (self.radius - w) * np.sin(a)]
            else: # left turn
                a = self.startpoint_angle + cover_angle
                ptL = [cx + (self.radius - w) * np.cos(a), cy + (self.radius - w) * np.sin(a)]
                ptR = [cx + (self.radius + w) * np.cos(a), cy + (self.radius + w) * np.sin(a)]
        else: # point not in this track unit
            raise RuntimeError('forward trip point not in this unit!')
        return ptL, ptR

    """ -- new added Nov. 26, 2023 -- """
    def trip_dist_to_custompose(self, trip, dist):
        ''' 
            input:  in-unit trip and centerline dist
            output: position and angle
        '''
        straight_trip = self.len1
        curve_trip = abs(self.angle) * self.radius
        if trip <= straight_trip: # point in straight
            x0 = dist
            y0 = trip
            # rotate and move to the unit-i
            cx, cy = self.zero_pos[0], self.zero_pos[1]
            a = self.rotate_angle
            ptx = x0 * np.cos(a) - y0 * np.sin(a) + cx
            pty = x0 * np.sin(a) + y0 * np.cos(a) + cy
            psi = adjust_angle(self.zero_angle)

        elif trip <= straight_trip + curve_trip: # point in curve
            covered_length = trip - straight_trip
            cover_angle = covered_length / self.radius
            if self.angle < 0: # right turn 
                a = self.startpoint_angle - cover_angle
                psi = self.zero_angle - cover_angle
            else: # left turn
                a = self.startpoint_angle + cover_angle
                psi = self.zero_angle + cover_angle
            cx, cy = self.curve_center[0], self.curve_center[1]
            r = self.radius + dist 
            ptx = cx + r * np.cos(a)
            pty = cy + r * np.sin(a)

            psi = adjust_angle(psi)

        else: # point not in this track unit
            raise RuntimeError('trip_dist_to_custompose: forward trip point not in this unit!')
        return ptx, pty, psi

    def draw_unit(self, fig_num):
        # activate track figure
        plt.figure(fig_num)

        # straight line part
        pt1, pt2 = self.inlineL1, self.inlineL2
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'k-')
        pt1, pt2 = self.inlineR1, self.inlineR2
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'k-')
        pt1, pt2 = self.inlineC1, self.inlineC2
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'k:')

        # curve part
        angle_interval = 0.5/180 * np.pi   # 0.5 deg
        angle_list = np.arange(self.angle_startpoint, self.angle_endpoint, angle_interval)
        cenx, ceny = self.curve_center[0], self.curve_center[1]
        ptLx, ptLy = self.curve_L * np.cos(angle_list) + cenx, self.curve_L * np.sin(angle_list) + ceny
        ptCx, ptCy = self.curve_C * np.cos(angle_list) + cenx, self.curve_C * np.sin(angle_list) + ceny
        ptRx, ptRy = self.curve_R * np.cos(angle_list) + cenx, self.curve_R * np.sin(angle_list) + ceny
        plt.plot(ptLx, ptLy, 'k-')
        plt.plot(ptCx, ptCy, 'k:')
        plt.plot(ptRx, ptRy, 'k-')
        # curve center
        # plt.plot(cenx, ceny, 'ro')
        # connect point
        # plt.plot(self.curve_out_C[0], self.curve_out_C[1], 'bo')


        
class SimpleTrackClass():

    def __init__(self):
        self.width = 20
        zero_pos = [0, 0]
        zero_angle = np.pi/2
        rotate_angle = 0
        start_trip = 0
        print(' ------- Track Unit 0 ------ ', )
        print('Pos:', zero_pos, 'Angle:', zero_angle/np.pi*180)
        self.unit0 = TrackUnitClass(width=20, 
                                    len1=130-24.689, 
                                    curve_angle=torad(-90), 
                                    radius=30, 
                                    zero_pos=zero_pos, 
                                    zero_angle=zero_angle,
                                    rotate_angle=rotate_angle,
                                    start_trip=start_trip)
        zero_pos, zero_angle, rotate_angle, start_trip = self.unit0.connect_info()

        print(' ------- Track Unit 1 ------ ', )
        print('Pos:', zero_pos, 'Angle:', zero_angle/np.pi*180)
        self.unit1 = TrackUnitClass(width=20, 
                                    len1=150, 
                                    curve_angle=torad(-150), 
                                    radius=15, 
                                    zero_pos=zero_pos, 
                                    zero_angle=zero_angle,
                                    rotate_angle=rotate_angle,
                                    start_trip=start_trip)
        zero_pos, zero_angle, rotate_angle, start_trip = self.unit1.connect_info()

        print(' ------- Track Unit 2 ------ ', )
        print('Pos:', zero_pos, 'Angle:', zero_angle/np.pi*180)
        self.unit2 = TrackUnitClass(width=20, 
                                    len1=120, 
                                    curve_angle=torad(150), 
                                    radius=20, 
                                    zero_pos=zero_pos, 
                                    zero_angle=zero_angle,
                                    rotate_angle=rotate_angle,
                                    start_trip=start_trip)
        zero_pos, zero_angle, rotate_angle, start_trip = self.unit2.connect_info()

        print(' ------- Track Unit 3 ------ ', )
        print('Pos:', zero_pos, 'Angle:', zero_angle/np.pi*180)
        self.unit3 = TrackUnitClass(width=20, 
                                    len1=86.423, 
                                    curve_angle=torad(-180), 
                                    radius=20, 
                                    zero_pos=zero_pos, 
                                    zero_angle=zero_angle,
                                    rotate_angle=rotate_angle,
                                    start_trip=start_trip)
        zero_pos, zero_angle, rotate_angle, start_trip = self.unit3.connect_info()

        print(' ------- Track Unit 4 ------ ', )
        print('Pos:', zero_pos, 'Angle:', zero_angle/np.pi*180)
        self.unit4 = TrackUnitClass(width=20, 
                                    len1=150, 
                                    curve_angle=torad(-90), 
                                    radius=30, 
                                    zero_pos=zero_pos, 
                                    zero_angle=zero_angle,
                                    rotate_angle=rotate_angle,
                                    start_trip=start_trip)
        zero_pos, zero_angle, rotate_angle, start_trip = self.unit4.connect_info()

        print('---------- Track End -----------')
        print('Pos:', zero_pos, 'Angle:', zero_angle/np.pi*180)
        print('---------------------- Track Initialized ----------------------')

        self.unit_list = [self.unit0, self.unit1, self.unit2, self.unit3, self.unit4]
        self.start_trip_list = [unit.start_trip for unit in self.unit_list]
        # print(self.start_trip_list)
        self.end_trip_list = [unit.end_trip for unit in self.unit_list]
        self.total_trip = self.end_trip_list[-1]
        print(self.end_trip_list)

    def findcar(self, pos):
        for i, unit in enumerate(self.unit_list):
            if unit.findcar(pos):
                self.centerlinepoint = unit.centerlinepoint
                self.centerlinedist  = unit.centerlinedist
                self.track_dir = unit.track_dir
                self.car_trip = unit.start_trip + unit.unit_trip
                self.car_in_unit = i 
                # print('find car in unit', i, '!')
                # print('dist:', self.centerlinedist)
                # car found in track, return
                return True
        # after search, not found, not in track
        self.car_in_unit = None
        self.centerlinepoint = None
        self.centerlinedist = None
        self.track_dir = None
        print('car out of track!')
        return False

    def find_forward_point_old(self, addtrip):
        ''' return the point in front of the car with a certain distance '''
        point_all_trip = self.car_trip + addtrip
        i = self.car_in_unit
        if point_all_trip < self.unit_list[i].end_trip: # inside the same unit
            unit_addtrip = point_all_trip - self.unit_list[i].start_trip
            pt = self.unit_list[i].trip_to_centerlinepoint(unit_addtrip)
            self.fpoint_in_unit = i
        else:  # in the next unit
            unit_addtrip = point_all_trip - self.unit_list[i].end_trip  # more trip in next unit
            if i < 4:  # not the last unit
                pt = self.unit_list[i+1].trip_to_centerlinepoint(unit_addtrip)
                self.fpoint_in_unit = i+1
            else:  # in the last unit
                pt = self.unit_list[0].trip_to_centerlinepoint(unit_addtrip)
                self.fpoint_in_unit = 0
        self.fpoint_unit_trip = unit_addtrip  # add trip in unit
        return pt

    def find_forward_point(self, addtrip):
        ''' return the point in front of the car by any distance '''
        point_all_trip = self.car_trip + addtrip
        i = self.car_in_unit
        temp_trip = self.car_trip + addtrip
        while temp_trip > self.unit_list[i].end_trip:
            # temp trip exceed current unit
            # then, move to next unit
            i += 1
            if i > 4:  # cross the start line
                temp_trip -= self.unit_list[4].end_trip
                i = 0
        # out of loop, means point in the unit of i
        unit_addtrip = temp_trip - self.unit_list[i].start_trip
        pt = self.unit_list[i].trip_to_centerlinepoint(unit_addtrip)
        self.fpoint_in_unit = i
        self.fpoint_unit_trip = unit_addtrip
        return pt

    def find_forward_edgepoint(self, addtrip):
        ''' find the edge point in front of the car by distance '''
        point_all_trip = self.car_trip + addtrip
        i = self.car_in_unit
        temp_trip = self.car_trip + addtrip
        while temp_trip > self.unit_list[i].end_trip:
            # temp trip exceed current unit
            # then, move to next unit
            i += 1
            if i > 4:  # cross the start line
                temp_trip -= self.unit_list[4].end_trip
                i = 0
        # out of loop, means point in the unit of i
        unit_addtrip = temp_trip - self.unit_list[i].start_trip
        ptL, ptR = self.unit_list[i].trip_to_edgelinepoint(unit_addtrip)
        self.fpoint_in_unit = i
        self.fpoint_unit_trip = unit_addtrip
        return ptL, ptR

    def find_cartrack_angle(self, pose):
        ''' find the angle between car heading and track direction '''
        psi = pose[2]
        return get_angle_diff(self.track_dir, psi)

    def find_forward_angle(self, pose, addtrip):
        ''' find the angle between car heading and the line of car--forwardpoint '''
        pt0 = [pose[0], pose[1]]
        heading_dir = pose[2]
        pt1 = self.find_forward_point(addtrip)
        pos_dir = get_bearing(pt0, pt1)
        forward_angle = pos_dir - heading_dir
        return adjust_angle(forward_angle)

    def find_forward_trackangle(self, pose, addtrip):
        ''' find angle between car heading and track direction of forward point '''
        pt0 = [pose[0], pose[1]]
        heading_dir = pose[2]
        pt1 = self.find_forward_point(addtrip)
        i = self.fpoint_in_unit
        forward_track_dir = self.unit_list[i].get_point_track_direction(pt1, self.fpoint_unit_trip)
        forward_trackangle = forward_track_dir - heading_dir
        return adjust_angle(forward_trackangle)

    def find_relative_edgepoint(self, pose, addtrip):
        pt0 = [pose[0], pose[1]]
        heading = pose[2] - np.pi/2 
        ptL0, ptR0 = self.find_forward_edgepoint(addtrip)
        ptL1 = [ptL0[0] - pt0[0], ptL0[1] - pt0[1]]
        ptR1 = [ptR0[0] - pt0[0], ptR0[1] - pt0[1]]
        # rotate to car heading direction
        def rotate(pt, a):
            x, y = pt[0], pt[1]
            x1 = x * np.cos(a) - y * np.sin(a)
            y1 = x * np.sin(a) + y * np.cos(a)
            return [x1, y1]
        ptL2 = rotate(ptL1, heading)
        ptR2 = rotate(ptR1, heading)
        return ptL2, ptR2

    def find_relative_centerpoint(self, pose, addtrip):
        pt0 = [pose[0], pose[1]]
        heading = -pose[2]
        cpt0 = self.find_forward_point(addtrip)
        pt1 = [cpt0[0] - pt0[0], cpt0[1] - pt0[1]]
        # rotate to car heading direction
        def rotate(pt, a):
            x, y = pt[0], pt[1]
            x1 = x * np.cos(a) - y * np.sin(a)
            y1 = x * np.sin(a) + y * np.cos(a)
            return [x1, y1]
        pt2 = rotate(pt1, heading)
        return pt2

    ''' --new added Nov. 29, 2023--'''
    def find_relative_front_oppo(self, pose, oppo_list):
        pt0 = [pose[0], pose[1]]
        heading = -pose[2]
        car_trip = self.car_trip
        min_trip_diff = self.total_trip
        min_trip_oppo = self.total_trip
        min_diff_index = None
        min_trip_index = None
        for i, oppo in enumerate(oppo_list):
            trip_diff = oppo.temp_trip - car_trip
            # find the nearest oppo in front
            if (trip_diff > 0) and (trip_diff < min_trip_diff):
                min_trip_diff = trip_diff
                min_diff_index = i
            # find the lowest oppo trip -- to deal with problem near start line
            if oppo.temp_trip < min_trip_oppo:
                min_trip_oppo = oppo.temp_trip
                min_trip_index = i

        if min_diff_index is None: # cannot find trip_diff > 0, means nearst front oppo is in a new lap
            front_oppo = oppo_list[min_trip_index]
        else:
            front_oppo = oppo_list[min_diff_index]
        # get position of front oppo
        oppo_pose = [front_oppo.pose[0], front_oppo.pose[1], front_oppo.pose[2]]
        # get relative vector 
        oppo_vector = [oppo_pose[0]-pt0[0], oppo_pose[1]-pt0[1]]
        oppo_spd = front_oppo.spd
        # rotate to car heading direction
        def rotate(pt, a):
            x, y = pt[0], pt[1]
            x1 = x * np.cos(a) - y * np.sin(a)
            y1 = x * np.sin(a) + y * np.cos(a)
            return [x1, y1]
        # relative vector
        pt2 = rotate(oppo_vector, heading)

        return pt2, oppo_spd, oppo_pose

    def random_car_pose(self):
        i = random.randint(0, 4)
        rate = 0.9    ##### used to be 0.8
        unit = self.unit_list[i]
        x0 = random.uniform(-self.width/2 * rate, self.width/2 * rate)
        y0 = random.uniform(0, unit.len1)    # used to be unit.len1/2
        # rotate and move to the place
        movex, movey =  unit.zero_pos[0], unit.zero_pos[1]
        a = unit.rotate_angle
        x1 = x0 * np.cos(a) - y0 * np.sin(a)
        y1 = x0 * np.sin(a) + y0 * np.cos(a)
        # move
        x = x1 + movex
        y = y1 + movey
        psi = adjust_angle(unit.zero_angle)
        return [x, y, psi]

    """ -- new added Nov. 26 2023 -- """
    def custom_car_pose(self, trip, dist):
        '''
        input:  trip and dist
        output: car pose, x, y, psi
        -- trip: from start line to custom position
        -- dist: distance from centerline
        '''
        if trip > self.unit_list[-1].end_trip:
            raise RuntimeError('custom_car_pose: set trip larger than track trip')
        # find the trip in which unit
        for i in range(5):
            if trip < self.unit_list[i].end_trip:
                unit = self.unit_list[i]
                unit_trip = trip - unit.start_trip
                break
        # put trip and dist to that Unit
        x, y, psi = unit.trip_dist_to_custompose(unit_trip, dist)

        return [x, y, psi]


    def show(self):
        fig_num = 'track'
        trackfig = plt.figure(num=fig_num, figsize=(7,7))
        # draw each track unit
        for unit in self.unit_list:
            unit.draw_unit(fig_num)
        # other units

        # draw function
        def draw_dir(x, y, a, c):
            plt.plot([x, x+np.cos(a)*5], [y, y+np.sin(a)*5], c)

        def draw_text(text):
            plt.text(0, -60, text)
        ''' Test positions '''

        ''' Test findcar function '''
        # pos1 = [5, 25]    # unit 1 line
        # pos2 = [8, 127]   # unit 1 curve
        # pos3 = [60, 132]    # unit 2 line
        # pos4 = [192, 127] # unit 2 curve
        # pos5 = [140, 83] # unit 3 line
        # pos6 = [72, 28]  # unit 3 curve
        # pos7 = [134, 10] # unit 4 line
        # pos8 = [204, -10.3] # unit 4 curve
        # pos9 = [113, -24] # unit 5 line
        # pos10 = [14, -15.7] # unit 5 curve
        # pos11 = [84.08789323914306, 46.4266461003934]  # test pose 1
        # pos12 = [69.89655791483735, 29.986686757847334]  # test pose 2
        # testlist = [pos1, pos2, pos3, pos4, pos5, pos6, pos8, pos9, pos10, pos11, pos12]
        # for pos in testlist:
        #     self.findcar(pos)
        #     plt.plot(pos[0], pos[1], 'r*')
        #     cpos = self.centerlinepoint
        #     plt.plot(cpos[0], cpos[1], 'g*')

        ''' Test find_forward_point function '''
        # pos = [0, 50]
        # self.findcar(pos)
        # plt.plot(pos[0], pos[1], 'r*')
        # for i in range(5):
        #     pt = self.find_forward_point((1+i)*20)
        #     plt.plot(pt[0], pt[1], 'g*')

        ''' Test find_forward_edgepoint function '''
        # pos = [0, 80]
        # self.findcar(pos)
        # plt.plot(pos[0], pos[1], 'ro')
        # for i in range(45):
        #     ptL, ptR = self.find_forward_edgepoint((i)*20)
        #     plt.plot(ptL[0], ptL[1], 'g*')
        #     plt.plot([pos[0], ptL[0]], [pos[1], ptL[1]], 'g:')
        #     plt.plot(ptR[0], ptR[1], 'b*')
        #     plt.plot([pos[0], ptR[0]], [pos[1], ptR[1]], 'b:')

        ''' test find_relative_edgepoint function '''
        # pos = [0, 80, np.pi/2+torad(10) ]
        # self.findcar(pos)
        # for i in range(10):
        #     ptL, ptR = self.find_forward_edgepoint((i)*10)
        #     plt.plot(ptL[0], ptL[1], 'g*')
        #     plt.plot([pos[0], ptL[0]], [pos[1], ptL[1]], 'g:')
        #     plt.plot(ptR[0], ptR[1], 'b*')
        #     plt.plot([pos[0], ptR[0]], [pos[1], ptR[1]], 'b:')

        # fig2 = plt.figure()

        # for i in range(10):
        #     ptL, ptR = self.find_relative_edgepoint(pos, (i)*10)
        #     ptL[0] /= 100
        #     ptL[1] /= 100
        #     ptR[0] /= 100
        #     ptR[1] /= 100
        #     plt.plot(ptL[0], ptL[1], 'g*')
        #     plt.plot(ptR[0], ptR[1], 'b*')

        ''' test find_relative_centerpoint function '''
        # pos = [0, 80, np.pi/2-torad(10) ]
        # self.findcar(pos)
        # for i in range(10):
        #     pt = self.find_forward_point((i+1)*10)
        #     plt.plot(pt[0], pt[1], 'g*')

        # fig2 = plt.figure()
        # for i in range(10):
        #     pt = self.find_relative_centerpoint(pos, (i+1)*10)
        #     pt[0] /= 100
        #     pt[1] /= 100
        #     plt.plot(pt[0], pt[1], 'g*')


        ''' Test find_cartrack_angle function '''
        # pose = [192, 127, torad(10)]
        # self.findcar(pose)
        # plt.plot(pose[0], pose[1], 'ro')  # car position
        # draw_dir(pose[0], pose[1], pose[2], 'r-')  # car heading
        # draw_dir(pose[0], pose[1], self.track_dir, 'k-' )  # track direction
        # cartrackangle = self.find_cartrack_angle(pose)
        # draw_text(str(todeg(cartrackangle))+' deg')

        ''' Test find_forward_angle function '''
        # pose = [60, 140, torad(20)]
        # self.findcar(pose)
        # plt.plot(pose[0], pose[1], 'ro')  # car position
        # draw_dir(pose[0], pose[1], pose[2], 'k-')  # car heading
        # # looking forward points
        # pt10 = self.find_forward_point(10)
        # pt20 = self.find_forward_point(20)
        # plt.plot(pt10[0], pt10[1], 'go')
        # plt.plot(pt20[0], pt20[1], 'go')
        # # looking forward line of sight
        # plt.plot([pose[0], pt10[0]], [pose[1], pt10[1]], 'g-')
        # plt.plot([pose[0], pt20[0]], [pose[1], pt20[1]], 'g-')
        # # looking angles
        # a10 = self.find_forward_angle(pose, 10)
        # a20 = self.find_forward_angle(pose, 20)
        # print('angle 10m:', todeg(a10))
        # print('angle 20m:', todeg(a20))
        # # car track angle
        # cartrackangle = self.find_cartrack_angle(pose)
        # print('cartrackangle:', todeg(cartrackangle))
        ''' Test find_forward_trackangle function '''
        # pose = [8, 127, torad(90)]
        # self.findcar(pose)
        # plt.plot(pose[0], pose[1], 'ro')  # car position
        # draw_dir(pose[0], pose[1], pose[2], 'k-')  # car heading
        # # looking forward points
        # pt10 = self.find_forward_point(10)
        # pt20 = self.find_forward_point(20)
        # plt.plot(pt10[0], pt10[1], 'go')
        # plt.plot(pt20[0], pt20[1], 'go')
        # # looking forward line of sight
        # plt.plot([pose[0], pt10[0]], [pose[1], pt10[1]], 'g-')
        # plt.plot([pose[0], pt20[0]], [pose[1], pt20[1]], 'g-')
        # # looking angles
        # a10 = self.find_forward_angle(pose, 10)
        # a20 = self.find_forward_angle(pose, 20)
        # print('angle 10m:', todeg(a10))
        # print('angle 20m:', todeg(a20))
        # # car track angle
        # cartrackangle = self.find_cartrack_angle(pose)
        # print('cartrackangle:', todeg(cartrackangle))
        # # forward car track angle
        # b10 = self.find_forward_trackangle(pose, 10)
        # b20 = self.find_forward_trackangle(pose, 20)
        # print('trackangle 10m:', todeg(b10))
        # print('trackangle 20m:', todeg(b20))

        ''' Test random_car_pose function '''
        # for i in range(10):
        #     pose = track.random_car_pose()
        #     plt.plot(pose[0], pose[1], 'ro')
        #     draw_dir(pose[0], pose[1], pose[2], 'k-')


        # save figure
        # trackfig.savefig('fulltrack.png')

        return trackfig


if __name__ == '__main__':
    track = SimpleTrackClass()
    track.show()
    plt.axis('equal')
    plt.show()


