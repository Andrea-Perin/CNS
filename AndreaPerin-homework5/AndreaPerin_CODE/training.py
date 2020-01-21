import dill
import numpy as np
import agent
import environment
import argparse 
import matplotlib.pyplot as plt
import os

if not os.path.exists('trajs'):
    os.makedirs('trajs')


parser = argparse.ArgumentParser(description='Input the parameters of the environment.')
parser.add_argument('--episodes', type=int, default=1000, 
                    help='The number of episodes.')
parser.add_argument('--eplen', type=int, default=50, 
                    help='The length of each episode.')
parser.add_argument('--gridsize', type=int, default=10, 
                    help='The number of tiles in the grid.')
parser.add_argument('--softmax', type=bool, default=False, 
                    help='Whether to use softmax when choosing actions.')
parser.add_argument('--sarsa', type=bool, default=False, 
                    help='Whether to use SARSA.')
args = parser.parse_args()

episodes = args.episodes         # number of training episodes
episode_length = args.eplen      # maximum episode length
x = args.gridsize                # horizontal size of the box
y = args.gridsize                # vertical size of the box
goal = [5, 9]                    # objective point
discount = 0.9                   # exponential discount factor
softmax = args.softmax           # set to true to use Softmax policy
sarsa = args.sarsa               # set to true to use the Sarsa algorithm
num_actions = 5                  # the total number of actions (in this case, U,D,L,R,Stay)


# alpha is set to be constant during the episodes
#alpha = np.ones(episodes) * 0.25  
alpha = np.array([5/t for t in range(1,episodes+1)])
# epsilon is power-law decreasing
epsilon = np.array([x**(-1.2) for x in range(1,episodes+1)])  

# initialize the agent
learner = agent.Agent(states = (x * y),
                      actions = num_actions,
                      discount = discount,
                      max_reward=1,
                      softmax=softmax,
                      sarsa=sarsa)

# intialize environment
env = environment.Environment(x,
                              y, 
                              goal = goal,
                              log=True)

# ENVIRONMENT SPECIFICATIONS
sand_patch = [[xs,ys] for xs in range(3,7) for ys in range(3,7)]
for s in sand_patch:
    env.add_sand(s)
env.add_sand([2,4])
env.add_sand([2,5])
env.add_sand([7,5])
env.add_sand([7,4])
env.add_sand([4,7])
env.add_sand([5,7])
env.add_sand([4,2])
env.add_sand([5,2])

# possible initializations
inits = env.get_inits()

# perform the training
for index in range(0, episodes):
    # to allow reproducibility, the starting point is the same for all plots
    if ((index+1)%50==0):
        initial = [4,0]
        state = initial
    else:
        # start from a random state
        initial_idx = np.random.randint(len(inits))
        initial = inits[initial_idx]
        state = initial
    # initialize agent position; also, new history
    env.set_start(state)
    # start training
    reward = 0
    # run episode
    for step in range(0, episode_length):
        # find state index
        state_index = state[0] * y + state[1]
        # choose an action
        action = learner.select_action(state_index, epsilon[index])
        # the agent moves in the environment
        result = env.move(action)
        # Q-learning update
        next_index = result[0][0] * y + result[0][1]
        learner.update(state = state_index,
                       action = action,
                       reward = result[1],
                       next_state = next_index,
                       alpha = alpha[index],
                       epsilon = epsilon[index])
        # update state and reward
        reward += result[1]
        state = result[0]
    reward /= episode_length
   
    
    # periodically save the agent
    if ((index + 1) % 10 == 0):
        with open('agent.obj', 'wb') as agent_file:
            dill.dump(agent, agent_file)
        print('Episode ', index + 1, 
              ': the agent has obtained an average reward of ', 
              reward, ' starting from position ', initial) 
    if ((index + 1) % 100 == 0):
        env.plot(traj=True)
        ax = plt.gca()
        ax.set_title('Reward: '+str(reward))
        plt.savefig('trajs/traj_'+'s_'*sarsa+'sft_'*softmax+str(index+1)+'.jpg',
                    bbox_inches='tight')
