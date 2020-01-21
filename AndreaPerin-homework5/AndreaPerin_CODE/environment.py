import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class Environment:

    state = []
    goal = []
    boundary = []
    action_map = {
        0: [0, 0],
        1: [0, 1],
        2: [0, -1],
        3: [1, 0],
        4: [-1, 0],
    }
    ### INITIALIZATION
    def __init__(self, x, y, goal, log=False):
        self.boundary = np.asarray([x, y])
        self.state = np.asarray([0,0])
        self.start = self.state
        self.goal = np.asarray(goal)
        self.wall = []
        self.sand = []
        self.log = log
        self.history = []
        
    def set_start(self,point):
        # choose a stating point and save it as such
        self.state = point
        self.start = self.state
        # erase the history and start anew if log=True
        if (self.log):
            del self.history[:]
            self.history = [np.asarray(self.state)]
        
    ### ENVIRONMENT MANIPULATION
    def add_wall(self, start, end):
        # vertical wall
        if (start[0]==end[0]):
            y1,y2 = min(start[1],end[1]), max(start[1],end[1])
            wall_pts = [[start[0],py] for py in range(y1,y2+1)]
        # horizontal wall
        if (start[1]==end[1]):
            x1,x2 = min(start[0],end[0]), max(start[0],end[0])
            wall_pts = [[px,start[1]] for px in range(x1,x2+1)]
        # non-straight walls are not implemented
        if (start[1]!=end[1] and start[0]!=end[0]):
            print('Invalid wall')
        # overwrite sand if it is the case
        for w in wall_pts:
            if (w in self.sand):
                self.sand.remove(w)
        # add walls to list of wall points
        self.wall.extend(wall_pts)
            
    def add_sand(self, point):
        if (point in self.wall):
            self.wall.remove(point)
        self.sand.append(point)
        
    def refresh(self):
        self.wall = []
        self.sand = []
        
    def get_inits(self):
        possible_inits = [[xi,yi] for 
                          xi in range(self.boundary[0]) for 
                          yi in range(self.boundary[1]) if 
                          ([xi,yi] not in self.wall) and 
                          ([xi,yi] not in self.sand) and
                          ([xi,yi]!=self.goal).all()]
        return possible_inits

    ### MOVEMENTS
    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def move(self, action):
        reward = 0
        movement = self.action_map[action]
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        if (self.check_sand(next_state)):
            reward = -0.5
        if (self.check_boundaries(next_state) or self.check_walls(next_state)):
            reward = -1
        else:
            self.state = next_state
        # if logging is on, keep track of moves
        if (self.log):
            self.history.append(self.state)
        return [self.state, reward]

    # map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0

    def check_walls(self, state):
        return list(state) in self.wall

    def check_sand(self, state):
        return list(state) in self.sand

    def plot(self,traj=False):
        plt.close('all')
        # define a matrix as the environment
        env = np.zeros((self.boundary[0],self.boundary[1]))
        for s in self.sand:
            env[s[0],s[1]] = -0.5
        for w in self.wall:
            env[w[0],w[1]] = -1.0
        env[self.goal[0],self.goal[1]] = 1.0
        env[self.start[0],self.start[1]] = 3.0        
        # define colors
        cmap = colors.ListedColormap(['black',   #walls
                                          '#c2b280', #sand
                                          'white',   #empty
                                          '#6f00ff',  #goal
                                          '#00ff00' #start
                                          ])
        bounds=[-2,-0.75,-0.1,0.1,2,2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        # plot everything
        fig, ax = plt.subplots()
        ax.tick_params(
                axis='both',
                which='both',
                bottom=False, 
                left=False, 
                top=False, 
                right=False,
                labelbottom=False,
                labelleft=False)
        ax.imshow(env.T,cmap=cmap, norm=norm, origin='lower')
        if (traj and self.log):
            ax.plot(*zip(*self.history), '-o')            
        return fig, ax

#%%
if __name__=="__main__":
    # intialize environment
    x=10
    y=10
    goal = [7,0]
    start = [0,0]
    env = Environment(x,
                      y, 
                      goal = goal)
    env.set_start(start)
    sand_patch = [[xs,0] for xs in range(1,7)]
    env.add_wall([6,0],[6,6])
    env.add_wall([8,0],[8,8])
    env.add_wall([3,1],[6,1])
    for s in sand_patch:
        env.add_sand(s)
    env.add_sand([2,1])
    env.plot()
    plt.show()