import numpy as np
import torch
import TD3
import utils
from ActionMapping import ActionMappingClass
from SimpleTrackEnv import SimpleTrackEnvClass

def process_state(s):
    return np.reshape(s, [1, -1])

# set parameters
dt = 0.01
state_dim = 31
action_dim = 2
max_action = 1
dt = 0.01

args = {
    'start_timesteps':1e4, 
    'eval_freq': 5e3,
    'expl_noise': 0.1, 
    'batch_size': 256,
    'discount': 0.99,
    'tau': 0.005,
    'policy_noise': 0.2,
    'noise_clip': 0.5,
    'policy_freq': 2   # was 2
}

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": args['discount'],
    "tau": args['tau'],
}

# Target policy smoothing is scaled wrt the action scale
kwargs["policy_noise"] = args['policy_noise'] * max_action
kwargs["noise_clip"] = args['noise_clip'] * max_action
kwargs["policy_freq"] = args['policy_freq']
policy = TD3.TD3(**kwargs)

replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(5e6))

# action mapping function 
am = ActionMappingClass()

env = SimpleTrackEnvClass()

# set training counters
stepcounter = 0
traincounter = 1
savecounter = 1
trainlog = []

for episode in range(10000):
    ob = env.reset()
    ob = process_state(ob)
    done = False
    saved = False
    episode_reward = 0
    episode_timesteps = 0
    for step in range(10000):
        time = step * dt
        stepcounter += 1

        # generate action for carA
        if stepcounter < args['start_timesteps']:
            action = np.random.uniform(-1, 1, action_dim)
        else:
            noise = np.random.normal(0, max_action * args['expl_noise'], size=action_dim)
            action = (policy.select_action(ob) + noise).clip(-max_action, max_action)  # clip here
        # action mapping 
        action_in = am.mapping(env.car.spd, env.car.steer, action[0], action[1])

        # perform action
        next_ob , r, done = env.step(action_in)
        next_ob = process_state(next_ob)  # convert to numpy.array
        # store replay buffer
        replay_buffer.add(ob, action, next_ob, r, done)

        # update state
        ob = next_ob
        episode_reward += r

        if done: break
        
        if stepcounter > args['start_timesteps']:
            policy.train(replay_buffer, args['batch_size'])
            traincounter += 1
        # save TD3 model
        if traincounter % 100000 == 0 and not saved:
            policy.save('models/model_'+str(savecounter))
            savecounter += 1
            saved = True
            print('TD3AM model', savecounter, 'saved!')
    # End of Episode
    fail_reason = env.query_fail_reason()
    print('episode: %d  reward: %.1f  step: %d counter: %d reason: %s' % (episode, episode_reward, step, traincounter, fail_reason))
    trainlog.append([episode, episode_reward, step, traincounter, fail_reason])
    # save reward log
    if saved:
        np.save('results/trainlog.npy', trainlog)