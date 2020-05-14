# WindyGridworld



<!-- SUMARIO -->
## SUMARIO

* [Introduccion](#Introduccion)
  * [Entorno](#Entorno)
* [Aprendizaje por refuerzo](#Aprendizaje-por-refuerzo)
  * [Estrategia](#Estrategia)
  * [Entrenamiento](#Entrenamiento)
  * [Algoritmos](#Algoritmos) 
    * [SARSA](#SARSA)
    * [Expected SARSA](#Expected-SARSA)
    * [Q Learning](#Q-Learning)
    * [Double Q learning](#Double-Q-learning)
* [Resultados](#Resultados)
  * [Evolucion](#Evolucion)
  * [Q Value](#Q-Value)
  * [Estrategia Óptima](#Estrategia-Óptima)
* [Conclusiones](#Conclusiones)
  * [SARSA vs Q-Learning](#SARSA-vs-Q-Learning)  
* [Anexo](#Anexo)
    * [SARSA](#SARSA)
    * [Expected SARSA](#Expected-SARSA)
    * [Q Learning](#Q-Learning)
    * [Double Q learning](#Double-Q-learning)




<!-- INTRODUCCION -->
## Introduccion

Este trabajo ha sido realizado como proyecto final del seminario "Reinforcement Machine Learning" de la ETSTB.
El objetivo principal es el de analizar y comprender diferentes algoritmos de *Time Difference Learning*. Para ser precisos, en este trabajo observaremos el comportamiento de los siguientes algritmos:

* SARSA
* Expected SARSA
* Q-Learning
* Double Q-Learning

Este estudio se hara en un entorno programado a medida en *python*.

### Entorno

El entorno se basa en un agente situado en una cuadrícula de 10 por 10 casillas. Cada casilla corresponde a un estado en el que nuestro agente se puede encontrar.
Se realizaran episodios que consistiran en lo siguiente:

1. El agente se situará en la casilla inicial
2. El agente escogerá una de entre cuatro acciones:
* Arriba
* Abajo
* Izquierda
* Derecha

Si la accion que toma el agente le lleva a salir de la cuadrícula, este no cambia de estado. 

3. Por cada acción que toma, el agente recibe una recompensa de -1
4. Si el agente llega a la casilla final, el episodio acaba

En este entorno, trayectos mas cortos entre la casilla inicial y final tendras recompensas totales mayores.
Además, en la cuadricula soplan vientos verticales y laterales, lo que provoca que el agente se vea desplazado por estos cada vez que realiza una accion. Estos se representan con una flecha si la intensidad del viento es 1, y con dos flechas si la intensidad del viento es 2.

Aqui podemos observar un ejemplo de un agente que ha seguido una ruta determinada:

![Ejemplo de entorno](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/Environment.PNG)

*Fig 1. Ejemplo de entorno*




<!-- Aprendizaje por refuerzo -->
## Aprendizaje por refuerzo

Nuestro agente tiene que aprender, a base de realizar episodios, a conseguir la mejor puntuacion posible. Esto conlleva que el agente siga una estrategia óptima que le dicte que accion tomar segun el estado en el que se encuentre.


### Estrategia
La estrategia que llevará a cabo nuestro agente vendrá derivada de una función Q(*estado,accion*), la cual asigna a cada par estado-accion un valor proporcional a la esperanza de la recompensa final si el agente toma esa accion en ese estado. 
#### Óptima
La estretegia óptima será aquella que para cada estado, escoga la acción que maximize la función Q:
```python
def getOptPol(Q):
    for state in range(0,Q.shape[0]):        
        policy[state]=np.argmax(Q[state,...]) 
```
Esta estrategia está bien si la funcion Q esta bien estimada, pero en caso contrario la estrategia distará mucho de la óptima.
#### E-Greedy
Otra estrategia posible es añadir un poco de aleatoriedad a la accion a tomar, para favorecer la exploracion de otros estados.
Esta estrategia esocoge para cada estado con probabilidad 1-epsilon la accion que maximza Q, y con probabilidad epsilon cualquiera de las otras acciones al azar:
```python
def e_greedy(state,Q,epsilon):
   
    rng = np.random.default_rng()
    if random.random() > epsilon:        
        action = np.argmax(Q[state,...])        
    else:
        action = rng.integers(0,3,endpoint=True)
    
    return action
```
Esta estrategia nos será muy útil en el entrenamiento, como veremos a continuación.

### Entrenamiento
Al empezar el entrenamiento, el agente desconoce la funcion Q, por lo que la tendrá que estimar.
Para ello, realizará una serie de episodios donde observara la recompensa obtenida en funcion de las acciones tomadas, e irá actualizando la función Q para maximizar la recompensa.
Durante cada episodio del entrenamiento, nuestro agente seguriá la estrategia e-greedy mencionada anteriormente, para ir explorando estados diferentes del óptimo y así poder estimar mejor Q.
El entrenamiento finalizará cuando tras un episodio la función Q varíe menos que un cierto unmbral o cuando el agente haya realizado un numero maximo de episodios.
### Algoritmos
A continuacion analizaremos 4 algoritmos de la familia de Algoritmos ***Time Difference Learning***.
Todos los agoritmos que veremos son esencialmente muy similares, siendo la única diferencia el cálculo del parámetro encargado de ir actualizando la funcion Q.
#### SARSA
SARSA, siglas para State,Action,Reward,State',Action', es un algoritmo que, tras realizar una accion en un determinado estado para llegar a un nuevo estado, actualiza la funcion Q(estado,accion) teniendo en cuenta que acción tomaría el agente en este nuevo estado si utilizara la misma estrategia usada durante el entrenamiento (e-greedy).
Aqui podemos observar un ejemplo de la aplicacion de este algoritmo para un entrenamiento de duración *episodes*:
```python
while episode < episodes: 

    #Inicializamos el estado
    state = agent.setState(gridworld.IniState,gridworld)  
    #Tomamos una accion (e-greedy) 
    action = e_greedy(state,Q,epsilon)   
    
    while state is not gridworld.FinalState and fitness[episode] > -1000:

        #Observamos a que estado llegamos
        sprime = gridworld.move(agent,action)
        #Observamos que accion tomariamos en ese estado
        aprime = e_greedy(sprime,Q,epsilon)
        #Actualizamos la función Q
        Q[state,action] = Q[state,action] + alpha*(gridworld.reward(state,action) + gamma*Q[sprime,aprime] - Q[state,action])
        
        state = sprime
        action = aprime  

    episode = episode + 1
```
#### Expected SARSA
Este algoritmo es una pequeña modificación de SARSA. La diferencia reside en que, una vez tomada una accion y viendo a que estado se llega, la función Q no se actualizaria teniendo solamente en cuenta la accion tomada segun la estrategia e-greedy, si no que realizaria un promedio para todas las acciones sobre el valor de la funcion Q en ese estado.
Aqui podemos observar un ejemplo de la aplicacion de este algoritmo para un entrenamiento de duración *episodes*:
```python
while episode < episodes: 

    #Inicializamos el estado
    state = agent.setState(gridworld.IniState,gridworld)  
    #Tomamos una accion (e-greedy) 
    action = e_greedy(state,Q,epsilon)   
    
    while state is not gridworld.FinalState and fitness[episode] > -1000:

        #Observamos a que estado llegamos
        sprime = gridworld.move(agent,action)
        #Actualizamos la función Q
        expectation = (1-epsilon)*Q[sprime,...].max() + epsilon*(Q[sprime,...].sum()-Q[sprime,...].max()) 
        Q[state,action] = Q[state,action] + alpha*(gridworld.reward(state,action) + gamma*expectation - Q[state,action])
        
        state = sprime
        action = aprime  

    episode = episode + 1
```
#### Q Learning
El algoritmo Q learning, a diferencia del SARSA, no utiliza la estrategia e-greedy para ver que accion tomaria en el siguiente estado. En su lugar utilzia la estrategia determinista óptima, donde la accion tomada será la que maximice Q(state').
Aqui podemos observar un ejemplo de la aplicacion de este algoritmo para un entrenamiento de duración *episodes*:
```python
while episode < episodes: 

    #Inicializamos el estado
    state = agent.setState(gridworld.IniState,gridworld)  
    #Tomamos una accion (e-greedy) 
    action = e_greedy(state,Q,epsilon)   
    
    while state is not gridworld.FinalState and fitness[episode] > -1000:

        #Observamos a que estado llegamos
        sprime = gridworld.move(agent,action)
        #Actualizamos la función Q
        Q[state,action] = Q[state,action] + alpha*(gridworld.reward(state,action) + gamma*Q[sprime,np.argmax(Q[sprime,...])] - Q[state,action])
        
        state = sprime
        action = aprime  

    episode = episode + 1
```
#### Double Q Learning
El algoritmo Double Q Learning intenta reducir el sesgo que se introduce en los demas algoritmos por utilizar la funcion Q para actualizarse a ella misma. Para solucionar esto, este algoritmo cuenta con dos funciones: Q1 y Q2. La función Q2 se utiliza en cada paso para actualizar la función Q1 y viceversa.
Aqui podemos observar un ejemplo de la aplicacion de este algoritmo para un entrenamiento de duración *episodes*:

```python
while episode < episodes: 

    #Inicializamos el estado
    state = agent.setState(gridworld.IniState,gridworld)  
    #Tomamos una accion (e-greedy) 
    action = e_greedy(state,Q,epsilon)   
    
    while state is not gridworld.FinalState and fitness[episode] > -1000:

        #Observamos a que estado llegamos
        sprime = gridworld.move(agent,action)        
        #Actualizamos las funciones Q1 y Q2
        if random.random() > 0.5:
                Q1[state,action] = Q1[state,action] + alpha*(gridworld.reward(state,action) + gamma*Q2[sprime,np.argmax(Q1[sprime,...])] - Q1[state,action])
            else:
                Q2[state,action] = Q2[state,action] + alpha*(gridworld.reward(state,action) + gamma*Q1[sprime,np.argmax(Q2[sprime,...])] - Q2[state,action])
        
        state = sprime
        action = aprime  

    episode = episode + 1
```
## Resultados
A continuación analizaremos los resultados obtenidos tras entrenar a nuestro agente con las siguientes condiciones:
* Cuadrícula de 10 por 10
* Estado inicial = 14
* Estado final = 65
* Viento vertical aleatorio entre 0 i 2
* Viento vertical aleatorio entre 0 i 1
* alpha = 0.3
* gamma = 0.3
* epsilon = 0.05
* Criterio de convergencia |Q-Q'| < 1e-20
### Evolucion
A continuación analizaremos la evolución del comportamiento del agente a lo largo del entrenamiento.
Para empezar veamos el primer episodio que realiza nuestro agente:
![First Run](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/First_run.png)

*Fig.2 Primer episodio*

Al principio del entrenamiento el agente actúa caoticamente dado que tiene una estrategia aleatoria. 
Veamos como se comporta el agente cuando llevamos un 10% del entrenamiento:
![First Run](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/First_run.png)

*Fig.3 Episodio al 10% del entrenamiento*

Nada mal. Nuestro agente ha aprendido que tiene que llegar a la casilla amarilla. Aun puede mejorarse, veamos como se comporta cuando termina su entrenamiento:
![First Run](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/First_run.png)

*Fig.4 Episodio final*

Podemos ver como el agente ha perfeccionado su estrategia con todos los algoritmos.

*Nota:Con algoritmos SARSA no hay diferencia entre el episodio intermedio y el final. Esto es debido a que, como veremos mas adelante (Fig.5), la funcion Q tarda mas en converger o directamente no converge, y por tanto el episodio mostrado (cuando el agente llevaba del 10% del entrenamiento) es un episodio mucho mas avanzado que en los otros casos*

A continuación se muestra la evolución tanto de la estimación de la función Q como del numero de pasos necesarios en cada episodio para llegar del estado inicial al estado final.
#### Convergencia de la función Q
En el eje vertical nos encontramos con el logaritmo en base 10 de la variación de la función Q entre episodios:
![Log Evolution](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/Logarithmic_evolution.png)

*Fig.5 Convergencia de Q*

Podemos observar como el algoritmo Q-Learning converge primero, antes de las 700 iteraciones. Le sigue el Double Q-Learning con el doble de iteraciones aproximadamente, lo cual tiene sentido ya que tiene que estimar 2 funciones Q en lugar de una. El Expected SARSA llega a converger aunque le cuesta casi 14000 iteraciones. El algoritmo SARSA no llega a converger. Este asunto se discutirá en las conclusiones.


#### Pasos por episodio
La siguiente figura muestra los pasos llevados a cabo por el agente para llegar del estado inicial al estado final:
![Step Evolution](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/α=0.3γ=0.3ε=0.05.png)

*Fig.6 Pasos por episodio*

Al principio del entrenamiento el agente toma decisiones al azar y como consecuencia el numero de pasos hasta llegar a la casilla amarilla es muy alto. En pocas iteraciones el agente aprende que la estrategia es llegar a la casilla amarilla, por tanto no se suele perder mucho y el numero de pasos por episodio se estabiliza.
### Q Value
La siguiente figura muestra el valor de la funcion Q para cada estado. Este valor es la suma de los valores en un esado Q(estado,accion) para todas las acciones.
Para los mas curiosos, en [Anexo](#Anexo) se pueden encontrar las figuras que muestran los valores de Q(estado) para cada accion por separado. No se incluyen aquí porque la información que aportan no es comoda de interpretar.
*Nota: En todos los estados (exclusivamente) que no han sido visitados durante el entrenamiento, la función Q vale 0*  

![Q Function](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/Q_Value.png)

*Fig.7 Q(estado)*

### Estrategia Óptima

Ahora nuestro agente conoce la función Q que el ha estimado. Esta función estará bien aproximada si el entrenamiento se ha realizado con exito. 
Teniendo esto en cuenta, lo mas inteligente será que ahora el agente lleve a cabo una estrategia determinista, realizando para cada estado la accion que maximiza Q.
Ahora observamos que accion tomaría nuestro agente para cada estado:

![Optimal Policy](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/Optimal_Policy.png)

*Fig.8 Estrategia óptima*

Si ahora dejamos que el agente siga esta estrategia durante un episodio, esto es lo que sucederia:

![Optimal Run](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/Optimal_Run.png)

*Fig.9 Episodio óptimo*

<!-- CONCLUSIONES -->
## Conclusiones
### SARSA vs Q Learning
Con los resultados que se han mostrado se puede concluir que, pese a llegar ambos algoritmos a soluciones identicas, Q Learning consigue que la función Q converga. SARSA por su parte no llega a este punto y la función Q nunca llega a converger (pese a que no varia demasiado de episodio a episodio).
Esto demuestra que, pese a llevar una estrategia e-greedy, utilizar una estrategia determinista para analizar el estado futuro es bastante mejor que volver a aplicar la misma estrategia e-greedy. 
Esto hace que Q-Learning analice el verdadero potencial del estado futuro al mirar el valor de Q para la accion que lo maximice, mientras que SARSA tenga una probabilidad de devolver un valor para ese estado futuro que no representa fielmente su potencial (ya que se escoge el valor de Q siguiendo una estrategia e-greedy).
## Anexo
Aqui podemos observar de manera desglosada la función Q(estado,accion) obtenida en cada uno de los algoritmos.
### SARSA
![Q SARSA](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/SARSA.png)

*Fig.10 Q(estado,acción) SARSA*
### Expected SARSA
![Q Expected SARSA](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/Expected_SARSA.png)

*Fig.11 Q(estado,acción) Expected SARSA*
### Q Learning
![Q Q Learning](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/Q-Learning.png)

*Fig.12 Q(estado,acción) Q-Learning*
### Double Q learning
![Q Double Q learning](https://github.com/pablomarinreyes/WindyGridworld/blob/master/images/Double_Q-Learning.png)

*Fig.13 Q(estado,acción) Double Q-Learning*