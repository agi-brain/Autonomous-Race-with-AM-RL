import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.animation as animation


def show_animi(env, logger, Tmax, save=True, path='video.mp4'):
    # merge two lap data
    posx_list  = logger.lap1_log.posx  + logger.lap2_log.posx
    posy_list  = logger.lap1_log.posy  + logger.lap2_log.posy
    psi_list   = logger.lap1_log.psi   + logger.lap2_log.psi
    spd_list   = logger.lap1_log.spd   + logger.lap2_log.spd
    steer_list = logger.lap1_log.steer + logger.lap2_log.steer
    ux_list    = logger.lap1_log.ux    + logger.lap2_log.ux
    T_list     = logger.lap1_log.T     + logger.lap2_log.T
    am_list    = logger.lap1_log.amflag+ logger.lap2_log.amflag

    # build track 
    trackfig = env.track.show()
    plt.figure('track')
    plt.axis('equal')
    ax = plt.gca()
    
    # build car rectangle, 
    # display size is 1.5 times larger than real car size
    amp = 1.5
    car_width = 1.8 * amp
    car_length = 4.8 * amp
    def build_rect_patch(posx, posy, psi, color='b'):
        rect = Rectangle(xy=(posx, posy), height=car_width, width=car_length, color=color)
        # reference vector: point to center of rectangle (vx, vy)
        vx, vy = car_length/2, car_width/2
        # rotate reference relative vector
        vx1 = vx*np.cos(psi) - vy*np.sin(psi)
        vy1 = vx*np.sin(psi) + vy*np.cos(psi)
        # first rotate rect patch angle in deg
        rect.set_angle(psi/np.pi*180)
        # move rectangle offset
        rect.set_xy((posx-vx1, posy-vy1))
        return rect
    # show main car rectangle
    car_patch = build_rect_patch(posx_list[0], posy_list[0], psi_list[0], color='r')
    ax.add_patch(car_patch)

    # display text on scene
    text1 = plt.text(0, 160, ' ')
    text2 = plt.text(0, 150, ' ')
    text3 = plt.text(50, 150, ' ')
    text4 = plt.text(50, 160, ' ')
    text5 = plt.text(160, 155, ' ')
    text6 = plt.text(100, 160, ' ')
    text7 = plt.text(150, 150, ' ')
    # show gas/brake and steer lines
    gaspos    = [120, 150]
    steerpos  = [120, 160]
    linelen   = 15
    gasback   = plt.plot([gaspos[0]-linelen, gaspos[0]+linelen], [gaspos[1], gaspos[1]], c='silver')
    steerback = plt.plot([steerpos[0]-linelen, steerpos[0]+linelen], [steerpos[1], steerpos[1]], c='silver')
    gasline,  = plt.plot(gaspos[0], gaspos[1], 'g-', linewidth=5)
    brkline,  = plt.plot(gaspos[0], gaspos[1], 'r-', linewidth=5)
    steerline, = plt.plot(steerpos[0], gaspos[1], 'b-', linewidth=5)

    def update_scene(i):
        def update_rect_patch(rect, posx, posy, psi):
            vx, vy = car_length/2, car_width/2
            vx1 = vx*np.cos(psi) - vy*np.sin(psi)
            vy1 = vx*np.sin(psi) + vy*np.cos(psi)
            rect.set_xy((posx, posy))
            rect.set_angle(psi/np.pi*180)
            rect.set_xy((posx-vx1, posy-vy1))

        # update car patch
        update_rect_patch(car_patch, posx_list[i], posy_list[i], psi_list[i])

        # update text
        steer_angle = steer_list[i] / np.pi * 180
        steer_prop  = steer_angle / 12.0
        text1.set_text('Time: %.1f' % T_list[i])
        text2.set_text('Spd : %.1f' % spd_list[i])
        text3.set_text('Gas/Brk: %.1f' % ux_list[i])
        text4.set_text('Steer  : %.1f' % steer_angle)
        # if AM works
        if am_list[i]:
            text5.set_text('AM')
            text5.set_color('red')
        else:
            text5.set_text('  ')
        
        # update steer line
        steerline.set_data([steerpos[0], steerpos[0]-steer_prop*linelen], [steerpos[1], steerpos[1]])
        # update gas/brake line
        if ux_list[i] > 0:
            brkline.set_data(gaspos[0], gaspos[1])
            gasline.set_data([gaspos[0], gaspos[0]+ux_list[i]*linelen], [gaspos[1], gaspos[1]])
        else:  # brake line
            gasline.set_data(gaspos[0], gaspos[1])
            brkline.set_data([gaspos[0], gaspos[0]+ux_list[i]*linelen], [gaspos[1], gaspos[1]])
            
        # gather anmi objects
        text_list = [text1, text2, text3, text4, text5]
        line_list = [brkline, gasline, steerline] 

        # return oppo_patch_list + [car_patch] + [oppo_vector] + text_list + line_list
        return [car_patch] + text_list + line_list

    ani = animation.FuncAnimation(trackfig, update_scene, np.arange(0,Tmax,5), interval=20, blit=True)
    # save mp4 file
    if save:
        ani.save(path, writer='ffmpeg', fps=30)
        print('video saved to ' + path)
    plt.show()   # this show function is necessary!!!



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