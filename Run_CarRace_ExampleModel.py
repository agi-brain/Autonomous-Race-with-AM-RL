import numpy as np
import torch
import TD3
import utils
from ActionMapping import ActionMappingClass
from SimpleTrackEnv import SimpleTrackEnvClass
from LogTools import CarStateLogClass
from RenderVideo import show_animi

def process_state(s):
    return np.reshape(s, [1, -1])

# load policy
kwargs = {
    "state_dim": 31,
    "action_dim": 2,
    "max_action": 1,
}
model_path = 'example_model/model_1969'
policy = TD3.TD3(**kwargs)
# policy.load() # run GPU saved model on GPU
policy.load_gpu2cpu(model_path) # run GPU saved model on CPU

# load action mapping function 
am = ActionMappingClass()
# load environment 
env = SimpleTrackEnvClass()

# init environment
episode_reward = 0
ob = env.test_reset()
ob = process_state(ob)
pre_trip = 0
lap1_done = False
logger = CarStateLogClass(env)

# start race
for step in range(10000):
    # generate action
    action = policy.select_action(ob)
    # apply action mapping
    action_in = am.mapping(env.car.spd, env.car.steer, action[0], action[1])
    # perform action
    next_ob , r, done = env.step(action_in)
    # update state
    ob = process_state(next_ob)
    # save reward
    episode_reward += r
    if done: break

    # count lap 1 and 2
    if env.track.car_trip  < pre_trip and not lap1_done:
        # record time for lap 1
        lap1_time = step
        print('lap 1 finished!  lap time:', lap1_time/100)
        lap1_done = True

    if env.track.car_trip < pre_trip and lap1_done:
        # record time for lap 2
        lap2_time = step - lap1_time
        if lap2_time > 1000:
            print('lap 2 finished!  lap time:', lap2_time/100)
            break
    pre_trip = env.track.car_trip
    # log 
    logger.log_data(step, lap1_done, action, action_in)
# end of race
fail_reason = env.query_fail_reason()
print('Race Done! reward: %.1f  step: %d reason: %s' % (episode_reward, step, fail_reason))

# Show Race results 
logger.show_trajectory(lap='lap1')
logger.show_trajectory(lap='lap2')

logger.show_states_controls(lap='lap1')
logger.show_states_controls(lap='lap2')

show_animi(env, logger, step, save=True, path='results/TrackA_Video2.mp4')
