import numpy as np
import random
import math
import TD

class gridworld:    

    def __init__(self,Nrows,Ncolumns,IniState,FinalState) :

        gridworld.rng = np.random.default_rng(12)
        gridworld.Nrows = Nrows
        gridworld.Ncolumns = Ncolumns
        gridworld.IniState = IniState
        gridworld.IniRow,gridworld.IniCol = self.statetoxy(IniState)
        gridworld.FinalState = FinalState
        gridworld.FinalRow,gridworld.FinalCol = self.statetoxy(FinalState)
        gridworld.WindyC = self.rng.integers(0,2,size=Ncolumns+1,endpoint=True)
        gridworld.WindyC[Ncolumns] = 0
        #gridworld.WindyC = np.zeros(Nrows)
        gridworld.WindyR = self.rng.integers(0,1,size=Nrows+1,endpoint=True)
        #gridworld.WindyR = np.zeros(Nrows+1)
        gridworld.WindyR[Nrows] = 0
        gridworld.Nactions = 4
        gridworld.Ns = Nrows*Ncolumns
        gridworld.grid = np.zeros((Nrows+1,Ncolumns+1))
        gridworld.grid[self.IniRow][self.IniCol] = 3
        gridworld.grid[self.FinalRow][self.FinalCol] = 4
        gridworld.grid[...,Nrows]=self.WindyR
        gridworld.grid[Ncolumns]=self.WindyC

    def move(self,agent,action):

        if action == 0:
            rowp = agent.row - 1 - self.WindyC[agent.col]
            colp = agent.col - self.WindyR[agent.row]
                            
        elif action == 1:
            colp = agent.col + 1 - self.WindyR[agent.row]
            rowp = agent.row - self.WindyC[agent.col]
            
        elif action == 2:
            rowp = agent.row + 1 - self.WindyC[agent.col]
            colp = agent.col - self.WindyR[agent.row]
                            
        elif action == 3:
            colp = agent.col - 1 - self.WindyR[agent.row]
            rowp = agent.row - self.WindyC[agent.col]
            

        if rowp < 0: rowp = 0
        if rowp > self.Nrows - 1: rowp = self.Nrows - 1
        if colp < 0: colp = 0
        if colp > self.Ncolumns - 1: colp = self.Ncolumns - 1
        
        agent.row = int(rowp)
        agent.col = int(colp)        
        agent.state = self.xytostate(agent.row,agent.col)

        return agent.state
    
    def statetoxy(self,state):
        col = math.floor((state ) / gridworld.Ncolumns)
        row = (state) % (gridworld.Ncolumns)
        return row,col
        
    def xytostate(self,row,col):
        state = int((col)*self.Nrows + row)
        return state

    def reward(self,state,action) :
        return -1

    def episode(self, agent, policy):
        state = agent.restart(self) 
        run = []
        a = []
        fitness = 0
        run.append(agent.state)  
        while state is not gridworld.FinalState and fitness > -200:
            action = policy[state]
            sprime = self.move(agent,action)
            a.append(action)
            run.append(sprime) 
           
            fitness=fitness + self.reward(state,action)
            state = sprime
        return run,a
        # TD.show_run(run,a,self,"Run of "+agent.name)

class agent:    
     
    def __init__(self,gridworld,state):
        agent.state = state
        agent.row,agent.col = gridworld.statetoxy(state)
        agent.name = "Agent 007"
        

    def setState(self,state,gridworld):
        self.state = state
        self.row,self.col = gridworld.statetoxy(state)
        return state
    
    def restart(self,gridworld):
        self.state = gridworld.IniState
        self.row,self.col = gridworld.statetoxy(self.state)
        return self.state



