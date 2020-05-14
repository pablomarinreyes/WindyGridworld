import numpy as np
import random
import math

class gridworld:    

    def __init__(self,Nrows,Ncolumns,IniState,FinalState) :

        gridworld.rng = np.random.default_rng()
        gridworld.Nrows = Nrows
        gridworld.Ncolumns = Ncolumns
        gridworld.IniState = IniState
        gridworld.FinalState = FinalState
        gridworld.WindyC = self.rng.integers(0,2,size=Ncolumns,endpoint=True)
        #gridworld.WindyR = self.rng.integers(0,2,size=Nrows,endpoint=True)
        gridworld.WindyR = np.zeros(Nrows)
        gridworld.Nactions = 4

    def move(self,agent,action):

        if action == 0:
            rowp = agent.row - 1 - self.WindyC[agent.col]
            colp = agent.col - self.WindyR[agent.row]
                            
        elif action == 1:
            colp = agent.col + 1 - self.WindyR[agent.row]
            rowp = agent.row - self.WindyC[agent.col]
            
        elif action == 2:
            colp = agent.col - 1 - self.WindyR[agent.row]
            rowp = agent.row - self.WindyC[agent.col]
                            
        elif action == 3:
            rowp = agent.row + 1 - self.WindyC[agent.col]
            colp = agent.col - self.WindyR[agent.row]

        if rowp < 1: rowp = 1
        if rowp > self.Nrows: rowp = self.Nrows
        if colp < 1: colp = 1
        if colp > self.Ncolumns: colp = self.Ncolumns
        
        agent.row = rowp
        agent.col = colp        
        agent.state = agent.col*self.Nrows + agent.row - 1

        return agent.state
        
    
    def reward(self,state,action) :
        return -1

class agent:    
     
    def __init__(self,gridworld,state):
        agent.state = state
        agent.col = math.ceil((state + 1) / gridworld.Nrows)
        agent.row = (state + 1) % gridworld.Nrows



