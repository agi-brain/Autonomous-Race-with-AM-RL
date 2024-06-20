import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class LaplogClass():
    def __init__(self):
        self.posx = []
        self.posy = []
        self.psi = []
        self.spd = []
        self.steer = []
        self.ax = []
        self.ay = []
        self.ux = []
        self.uy = []
        self.T = []
        self.acc = []
        self.trip = []
        self.amflag = []

class CarStateLogClass():
    def __init__(self, env):
        self.env = env
        self.lap_flag = []
        self.lap_index = 1
        self.lap1_log = LaplogClass()
        self.lap2_log = LaplogClass()
        self.lap1_time = None
        self.lap2_time = None

    
    def log_data(self, step, lap1done, a_v, a_r):
        env = self.env
        if lap1done and not self.lap_flag[-1]:
            self.lap_index = 2
            self.lap1_time = len(self.lap_flag)
            # print('lap2 start at ', self.lap1_time, 'trip:', env.track.car_trip)
        self.lap_flag.append(lap1done)

        # select lap log first
        if self.lap_index == 1:
            log = self.lap1_log
        else:
            log = self.lap2_log
        
        log.T.append(step * 0.01)
        log.posx.append(env.car.pose[0])
        log.posy.append(env.car.pose[1])
        log.psi.append(env.car.pose[2])
        log.spd.append(env.car.spd)
        log.steer.append(env.car.steer)
        log.trip.append(env.track.car_trip)
        log.acc.append(env.car.acc_sum)

        log.ax.append(a_v[0])
        log.ay.append(a_v[1])
        log.ux.append(a_r[0])
        log.uy.append(a_r[1])

        # check if am working
        if a_v[0] != a_r[0] or a_v[1] != a_r[1]:
            log.amflag.append(True)
        else:
            log.amflag.append(False)


    # show trajectory
    def show_trajectory(self, lap='lap1'):
        if lap == 'lap1':
            laplog = self.lap1_log
            lap_time = self.lap1_time
        elif lap == 'lap2':
            if self.lap_index == 1:
                raise(RuntimeError('No lap2 log data found'))
            lap_time = (len(self.lap_flag) - self.lap1_time)/100
            laplog = self.lap2_log
        else:
            raise(ValueError('lap must be string: lap1 or lap2'))

        trackfig = self.env.track.show()
        fig1 = plt.figure('track', figsize=(4,4))
        norm = matplotlib.colors.Normalize(vmin=20,vmax=110)
        plt.scatter(laplog.posx, laplog.posy, c=np.array(laplog.spd)*3.6, cmap='jet', s=5, norm=norm)
        plt.colorbar(orientation='horizontal', shrink=0.7, pad=0)
        plt.plot([-15,15], [0,0], lw=3, c='grey')
        plt.axis('equal')
        plt.axis('off')
        # give title, lap1 time or lap2 time
        if lap_time is not None:
            lap_time_str = str(round(lap_time/100, 2)) + 's'
        else:
            lap_time_str = 'DNF'
        title_str = 'Track-A Trajectory ' + lap + ' (Lap Time =' + lap_time_str + ')'
        plt.title(title_str, size=11)
        plt.tight_layout()
        fig1.savefig('results/TrackA_Trajectory_' + lap + '.png', dpi=300)
        print('figure saved to results/TrackA_Trajectory_' + lap + '.png')
        plt.cla()
        plt.close()


    def show_states_controls(self, lap='lap1'):
        if lap == 'lap1':
            laplog = self.lap1_log
        elif lap == 'lap2':
            if self.lap_index == 1:
                raise(RuntimeError('No lap2 log data found'))
            laplog = self.lap2_log
        else:
            raise(ValueError('lap must be string: lap1 or lap2'))
        fig2, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(10,8))
        c_blue   = '#006699'
        # spd 
        ax1.plot(laplog.trip, np.array(laplog.spd)*3.6, c=c_blue)
        ax1.set_ylim([20,120])
        ax1.set_ylabel('Speed (km/h)')
        ax1.set_yticks(range(20,140,20))
        ax1.grid(axis='y', ls=':')
        ax1.set_title('Track-A States and Controls ' + lap, size=11)

        # acc/dec
        ax2.plot(laplog.trip, smooth(laplog.ux), c=c_blue)
        ax2.set_ylim([-1.25, 1.25])
        ax2.set_ylabel('Throttle/Brake')
        ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax2.grid(axis='y', ls=':')

        # steer 
        ax3.plot(laplog.trip, np.array(laplog.steer)/np.pi*180, c=c_blue)
        ax3.set_ylim([-10, 10])
        ax3.set_ylabel('Steer Angle (deg)')
        ax3.set_yticks([-10, -5, 0, 5, 10])
        ax3.grid(axis='y', ls=':')

        # acc
        ax4.plot(laplog.trip, smooth(laplog.acc)/9.81, c=c_blue)
        ax4.set_ylabel('Acceleration (g)')
        ax4.set_xlim([0,860])
        ax4.set_ylim([0, 1.3])
        ax4.set_xlabel('Track Distance (m)')
        ax4.set_yticks([0, 0.25, 0.5, 0.75, 1, 1.2])
        # ax4.grid(axis='y', ls=':')

        plt.tight_layout()

        fig2.savefig('results/TrackA_States_'+lap+'.png', dpi=300)
        print('figure saved to '+'results/TrackA_States_'+lap+'.png')
        plt.cla()
        plt.close()


def smooth(data):
    out = []
    window = 20
    out.append(data[0])
    for i in range(1, window-1):
        out.append(np.mean(data[0:i]))
    for i in range(window, len(data)):
        out.append(np.mean(data[i-window:i]))
    out.append(data[-1])
    return np.array(out)