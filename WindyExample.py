import environment as env
import TD
import numpy as np
import matplotlib.pyplot as plt

Nrows = 10
Ncolumns = 10
Ns = Nrows*Ncolumns
Na = 4
IniState = 14
FinalState = 65


gw = env.gridworld(Nrows,Ncolumns,IniState,FinalState)
agent = env.agent(gw,IniState)

episodes = 100000
lr = 0.3
depth = 0.3
epsilon = 0.05
criteria = 1e-20

Q_SARSA,fitness_SARSA,SARSA_behavoir,SARSA_choices,SARSA_evo = TD.SARSA(gw,agent,episodes,lr,depth,epsilon,criteria)
Q_ex_SARSA,fitness_ex_SARSA,ex_SARSA_behavoir,ex_SARSA_choices,ex_SARSA_evo = TD.expected_SARSA(gw,agent,episodes,lr,depth,epsilon,criteria)
Q_q,fitness_q,q_behavoir,q_choices,q_evo = TD.Q_learning(gw,agent,episodes,lr,depth,epsilon,criteria)
Q_2q1,Q_2q2,fitness_2q,Q2_behavoir,Q2_choices,Q2_evo = TD.DoubleQ_learning(gw,agent,episodes,lr,depth,epsilon,criteria)


policy_SARSA = TD.getOptPol(Q_SARSA)
policy_ex_SARSA = TD.getOptPol(Q_ex_SARSA)
policy_Q = TD.getOptPol(Q_q)
policy_2Q = TD.getOptPol(Q_2q1+Q_2q2)

Qp = np.array([Q_SARSA,Q_ex_SARSA,Q_q,Q_2q1+Q_2q2])
Bp = np.array([SARSA_behavoir,ex_SARSA_behavoir,q_behavoir,Q2_behavoir])
Cp = np.array([SARSA_choices,ex_SARSA_choices,q_choices,Q2_choices])
Fp = np.array([fitness_SARSA,fitness_ex_SARSA,fitness_q,fitness_2q])
Pp = np.array([policy_SARSA,policy_ex_SARSA,policy_Q,policy_2Q])
names = np.array(["SARSA","Expected SARSA","Q-Learning","Double Q-Learning"])

TD.show_results(Qp,Bp,Cp,Pp,Fp,names,gw,agent)







fig, axs = plt.subplots(2, 2)
fig.canvas.set_window_title('\u03B1	 = ' + str(lr) +' \u03B3 = ' + str(depth) + ' \u03B5 = ' + str(epsilon)) 
axs[0, 0].plot(range(0,fitness_SARSA.shape[0]), fitness_SARSA)
axs[0, 0].set_title('SARSA') 
axs[0, 1].plot(range(0,fitness_ex_SARSA.shape[0]), fitness_ex_SARSA, 'tab:orange')
axs[0, 1].set_title('Expected SARSA') 
axs[1, 0].plot(range(0,fitness_q.shape[0]), fitness_q, 'tab:green')
axs[1, 0].set_title('Q Learning') 
axs[1, 1].plot(range(0,fitness_2q.shape[0]), fitness_2q, 'tab:red')
axs[1, 1].set_title('Double Q Learning') 

for ax in axs.flat:
    ax.set(xlabel='Iteration', ylabel='Fitness')



fig, axs = plt.subplots(2,2)
fig.canvas.set_window_title("Logarithmic evolution") 
axs[0,0].set_title("SARSA")
axs[0,0].plot(range(0,SARSA_evo.shape[0]),np.log10(SARSA_evo+criteria))

axs[0,1].set_title("Expected SARSA")
axs[0,1].plot(range(0,ex_SARSA_evo.shape[0]),np.log10(ex_SARSA_evo + criteria))

axs[1,0].set_title("Q-Learning")
axs[1,0].plot(range(0,q_evo.shape[0]),np.log10(q_evo+criteria))

axs[1,1].set_title("Double Q-Learning ")
axs[1,1].plot(range(0,Q2_evo.shape[0]),np.log10(Q2_evo+criteria))

for ax in axs.flat:
    ax.set(xlabel='Iteration', ylabel='Q difference')



plt.show()