import environment as env
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def SARSA(gridworld,agent,episodes,lr,depth,epsilon,criteria):
    print("SARSA started")
    Q = np.zeros([gridworld.Nrows*gridworld.Ncolumns,gridworld.Nactions])
    fitness = np.array(())
    evolution = np.array(())
    behavoir = []
    choices = []
    episode = 0
    dif = 1e7
    while dif > criteria and episode < episodes:         
        state = agent.setState(gridworld.IniState,gridworld)
        run = []
        a = []
        run.append(agent.state) 
        action = e_greedy(state,Q,epsilon)
        aux = abs(Q).sum()
        fitness = np.append(fitness,0)
        while state is not gridworld.FinalState and fitness[episode] > -1000:
            sprime = gridworld.move(agent,action)
            aprime = e_greedy(sprime,Q,epsilon)

            a.append(action)
            run.append(sprime) 
            
            Q[state,action] = Q[state,action] + lr*(gridworld.reward(state,action) + depth*Q[sprime,aprime] - Q[state,action])
            fitness[episode]=fitness[episode] + gridworld.reward(state,action)
            state = sprime
            action = aprime
        
        dif=abs(aux-abs(Q).sum())
        evolution = np.append(evolution,dif)        

        a.append(-1)
        choices.append(a)
        behavoir.append(run)
        episode = episode + 1
        
    print("SARSA finished")
    return Q,fitness,behavoir,choices,evolution

def expected_SARSA(gridworld,agent,episodes,lr,depth,epsilon,criteria):
    print("Expected SARSA started")
    Q = np.zeros([gridworld.Nrows*gridworld.Ncolumns,gridworld.Nactions])
    fitness = np.array(())
    evolution = np.array(())
    behavoir = []
    choices = []
    episode = 0
    dif = 1e7
    while dif > criteria and episode < episodes:        
        state = agent.setState(gridworld.IniState,gridworld) 
        run = []
        a = []
        run.append(agent.state)  
        action = e_greedy(state,Q,epsilon)
        aux = abs(Q).sum()
        fitness = np.append(fitness,0)
        while state is not gridworld.FinalState and fitness[episode] > -1000:

            sprime = gridworld.move(agent,action)
            aprime = e_greedy(sprime,Q,epsilon)

            a.append(action)
            run.append(sprime) 

            expectation = (1-epsilon)*Q[sprime,...].max() + epsilon*(Q[sprime,...].sum()-Q[sprime,...].max()) 
            Q[state,action] = Q[state,action] + lr*(gridworld.reward(state,action) + depth*expectation - Q[state,action])
            fitness[episode]=fitness[episode] + gridworld.reward(state,action)
            state = sprime
            action = aprime
        dif=abs(aux-abs(Q).sum())
        evolution = np.append(evolution,dif)        
        
        a.append(-1)
        choices.append(a)
        behavoir.append(run)
        episode = episode + 1
    print("Expected SARSA finished")
    return Q,fitness,behavoir,choices,evolution

def Q_learning(gridworld,agent,episodes,lr,depth,epsilon,criteria):
    print("Q-Learning started")
    Q = np.zeros([gridworld.Nrows*gridworld.Ncolumns,gridworld.Nactions])
    fitness = np.array(())
    evolution = np.array(())
    behavoir = []
    choices = []
    episode = 0
    dif = 1e7
    while dif > criteria and episode < episodes:
        state = agent.setState(gridworld.IniState,gridworld) 
        run = []
        a = []
        run.append(agent.state)
        aux = abs(Q).sum()
        fitness = np.append(fitness,0)       
        while state is not gridworld.FinalState and fitness[episode] > -1000:

            action = e_greedy(state,Q,epsilon)
            sprime = gridworld.move(agent,action)

            a.append(action)
            run.append(sprime)          
            Q[state,action] = Q[state,action] + lr*(gridworld.reward(state,action) + depth*Q[sprime,np.argmax(Q[sprime,...])] - Q[state,action])
            fitness[episode]=fitness[episode] + gridworld.reward(state,action)
            state = sprime
        dif=abs(aux-abs(Q).sum())
        evolution = np.append(evolution,dif)        
        
        a.append(-1)
        choices.append(a)
        behavoir.append(run)
        episode = episode + 1
       
        
    print("Q-Learning finished")            
    return Q,fitness,behavoir,choices,evolution

def DoubleQ_learning(gridworld,agent,episodes,lr,depth,epsilon,criteria):
    print("Double Q-Learning started")
    Q1 = np.zeros([gridworld.Nrows*gridworld.Ncolumns,gridworld.Nactions])
    Q2 = np.zeros([gridworld.Nrows*gridworld.Ncolumns,gridworld.Nactions])    
   
    fitness = np.array(())
    evolution = np.array(())
    behavoir = []
    choices = []
    episode = 0
    dif = 1e7
    while dif > criteria and episode < episodes:    
        state = agent.setState(gridworld.IniState,gridworld) 
        run = []
        a = []
        run.append(agent.state)
        aux = abs(Q1+Q2).sum()
        fitness = np.append(fitness,0)  
        while state is not gridworld.FinalState and fitness[episode] > -1000:
            action = e_greedy(state,Q1+Q2,epsilon)
            sprime = gridworld.move(agent,action)
            a.append(action)
            run.append(sprime)  
            if random.random() > 0.5:
                Q1[state,action] = Q1[state,action] + lr*(gridworld.reward(state,action) + depth*Q2[sprime,np.argmax(Q1[sprime,...])] - Q1[state,action])
            else:
                Q2[state,action] = Q2[state,action] + lr*(gridworld.reward(state,action) + depth*Q1[sprime,np.argmax(Q2[sprime,...])] - Q2[state,action])            
            fitness[episode]=fitness[episode] + gridworld.reward(state,action)
            state = sprime
           
        
        dif=abs(aux-abs(Q1+Q2).sum())
        evolution = np.append(evolution,dif)        
        
        a.append(-1)
        choices.append(a)
        behavoir.append(run)
        episode = episode + 1        
    print("Double Q-Learning finished")
    return Q1,Q2,fitness,behavoir,choices,evolution

       
def e_greedy(state,Q,epsilon):
   
    rng = np.random.default_rng()
    if random.random() > epsilon:
        
        action = np.argmax(Q[state,...])
        
    else:
        action = rng.integers(0,3,endpoint=True)
    
    return action

def getOptPol(Q):
    policy = np.zeros(Q.shape[0])
    for state in range(0,Q.shape[0]): 
        if Q[state,...].sum() != 0:
            policy[state]=np.argmax(Q[state,...]) 
        else:
            policy[state]=-1
    return policy 

def show_policy(policy,gridworld,name,ax):        
    
    
    ax.set_title(name)
    ax.imshow(gridworld.grid)


    for i in range(0,gridworld.Nrows+1):
        y = gridworld.Ncolumns
        x = i 
       
        if gridworld.WindyC[i] == 1:
            Arrow = patches.Arrow(x,y+0.2,0,-0.4,width=0.5,color="white")
            ax.add_patch(Arrow)
        if gridworld.WindyC[i] == 2:
            Arrow1 = patches.Arrow(x-0.2,y+0.2,0,-0.4,width=0.5,color="white")
            Arrow2 = patches.Arrow(x+0.2,y+0.2,0,-0.4,width=0.5,color="white")        
            ax.add_patch(Arrow1)
            ax.add_patch(Arrow2)

    for i in range(0,gridworld.Ncolumns+1):
        x = gridworld.Nrows
        y = i 
       
        if gridworld.WindyR[i] == 1:
            Arrow = patches.Arrow(x+0.2,y,-0.4,0,width=0.5,color="white")
            ax.add_patch(Arrow)
        if gridworld.WindyR[i] == 2:
            Arrow1 = patches.Arrow(x+0.2,y,-0.4,0,width=0.5,color="white")
            Arrow2 = patches.Arrow(x-0.2,y,-0.4,0,width=0.5,color="white")        
            ax.add_patch(Arrow1)
            ax.add_patch(Arrow2)

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for state in range(0,policy.shape[0]):
        if policy[state] != -1:
            x,y = gridworld.statetoxy(state)
            posx = y 
            posy = x 
            if policy[state] == 0.:
                dx = 0
                dy = -0.4
            if policy[state] == 1.:
                dx = 0.4
                dy = 0
            if policy[state] == 2.:
                dx = 0
                dy = 0.4
            if policy[state] == 3.:
                dx = -0.4
                dy = 0
            Arrow = patches.Arrow(posx,posy,dx,dy,width=0.5)
            ax.add_patch(Arrow)

    #ax.autoscale_view()
    #ax.invert_yaxis()
def show_run(run,choices,gridworld,name,ax):        
    
    
    ax.set_title(name)
    ax.imshow(gridworld.grid)
    ax.patch.set_color('green')

    for i in range(0,gridworld.Nrows+1):
        y = gridworld.Ncolumns
        x = i 
       
        if gridworld.WindyC[i] == 1:
            Arrow = patches.Arrow(x,y+0.2,0,-0.4,width=0.5,color="white")
            ax.add_patch(Arrow)
        if gridworld.WindyC[i] == 2:
            Arrow1 = patches.Arrow(x-0.2,y+0.2,0,-0.4,width=0.5,color="white")
            Arrow2 = patches.Arrow(x+0.2,y+0.2,0,-0.4,width=0.5,color="white")        
            ax.add_patch(Arrow1)
            ax.add_patch(Arrow2)

    
    for i in range(0,gridworld.Ncolumns+1):
        x = gridworld.Nrows
        y = i 
       
        if gridworld.WindyR[i] == 1:
            Arrow = patches.Arrow(x+0.2,y,-0.4,0,width=0.5,color="white")
            ax.add_patch(Arrow)
        if gridworld.WindyR[i] == 2:
            Arrow1 = patches.Arrow(x+0.2,y,-0.4,0,width=0.5,color="white")
            Arrow2 = patches.Arrow(x-0.2,y,-0.4,0,width=0.5,color="white")        
            ax.add_patch(Arrow1)
            ax.add_patch(Arrow2)

    ax.patch.set_color('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    for state,action in zip(run,choices):
        if action != -1:
            x,y = gridworld.statetoxy(state)
            
            posx = y 
            posy = x 
            if action == 0.:
                dx = 0
                dy = -0.4
            if action == 1.:
                dx = 0.4
                dy = 0
            if action == 2.:
                dx = 0
                dy = 0.4
            if action == 3.:
                dx = -0.4
                dy = 0
            Arrow = patches.Arrow(posx,posy,dx,dy,width=0.5)
            ax.add_patch(Arrow)

def show_results(Q,behavoir,choices,policy,fitness,name,gw,agent):

    fig_pol, axs_pol = plt.subplots(2,2,figsize=(5, 5))
    fig_pol.canvas.set_window_title("Optimal Policy")
    fig_run, axs_run = plt.subplots(2,2,figsize=(5, 5))
    fig_run.canvas.set_window_title("Optimal Run")
    fig_Q, axs_Q = plt.subplots(2,2,figsize=(5, 5))
    fig_Q.canvas.set_window_title("Q Value")
    
    


    for i in range(0,name.shape[0]):        
        run,choices = gw.episode(agent,policy[i]) 
        show_run(run,choices,gw,f"{name[i]}",axs_run[i%2,math.floor(i/2)])
        show_policy(policy[i],gw,f"{name[i]}",axs_pol[i%2,math.floor(i/2)])
        showQ(Q[i],gw,f"{name[i]}",axs_Q[i%2,math.floor(i/2)],fig_Q)
        showQsa(Q[i],gw,f"{name[i]}")
    #fig_Q.colorbar(axs_Q)
    """run,choices = gw.episode(agent,policy)
    show_run(run,choices,gw,f"Optimal run of {name}")
    show_policy(policy,gw,"Optimal Policy of "+name)"""
    
    

def showQsa(Q,gridworld,name):
    
    fig, axs = plt.subplots(2,2,figsize=(5, 5))
    fig.canvas.set_window_title(name)
    
    im1 = axs[0, 0].imshow(Q[...,0].reshape(gridworld.Nrows,gridworld.Ncolumns,order='F'))
    fig.colorbar(im1,ax = axs[0, 0])
    axs[0, 0].set_title('Up') 

    im2 = axs[0, 1].imshow(Q[...,1].reshape(gridworld.Nrows,gridworld.Ncolumns,order='F'))
    fig.colorbar(im2,ax = axs[0, 1])
    axs[0, 1].set_title('Right') 

    im3 = axs[1, 0].imshow(Q[...,2].reshape(gridworld.Nrows,gridworld.Ncolumns,order='F'))
    fig.colorbar(im3,ax = axs[1, 0])
    axs[1, 0].set_title('Down')

    im4 = axs[1, 1].imshow(Q[...,3].reshape(gridworld.Nrows,gridworld.Ncolumns,order='F'))
    fig.colorbar(im4,ax = axs[1, 1])
    axs[1, 1].set_title('Left')
    
def showQ(Q,gridworld,name,ax,fig):
   Qp = np.zeros((gridworld.Ns,1)) 
   for state in range(0,gridworld.Ns): Qp[state] = Q[state,:].sum()   
   
   ax.set_title(name)
   im = ax.imshow(Qp.reshape(gridworld.Nrows,gridworld.Ncolumns,order='F'))
   fig.colorbar(im,ax = ax)
   