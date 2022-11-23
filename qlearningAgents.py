# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import random,util,math

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - computeValueFromQValues
      - computeActionFromQValues
      - getQValue
      - getAction
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    self.QValues = util.Counter()

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we have never seen a state
      or the Q node value otherwise
    """
    "*** YOUR CODE HERE ***"
    return self.QValues[(state, action)]
    util.raiseNotDefined()


  def computeValueFromQValues(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    value = float("-inf")
    actions = self.getLegalActions(state)

    if not actions:
      return 0.0

    for action in actions:
      if value < self.getQValue(state, action):
        value = self.getQValue(state, action)
    return value

  def computeActionFromQValues(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    value = float("-inf")
    #tiene que ser una accion valida si no peta el test
    actions = self.getLegalActions(state)
    maxAction = None

    #si no hay nada no devuelvo nada
    if not actions:
      return None

    #repasamos todas las acciones
    for action in actions:
      #comprobamos el valor hasta que encuentre el ideoneo 
      if value < self.getQValue(state, action):
        value = self.getQValue(state, action)
        maxAction = action

    #devuelvo la mejor accion
    return maxAction

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    "*** YOUR CODE HERE ***"
    #accion random pero legal
    randomAction = random.choice(legalActions)
    #cogemos la mejor accion calculada anteriormente
    bestAction = self.computeActionFromQValues(state)

    #usamos el hint para poder coger la accion
    if util.flipCoin(self.epsilon):
      action = randomAction
    else:
      action = bestAction

    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    originalValue = self.getQValue(state, action)
    nextStateValue = self.computeValueFromQValues(nextState)
    #aplicamos la formula
    newValue = (1 - self.alpha) * originalValue + self.alpha * (reward + self.discount * nextStateValue)
    self.QValues[(state, action)] = newValue

  def getPolicy(self, state):
    return self.computeActionFromQValues(state)

  def getValue(self, state):
    return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
  """
      ApproximateQLearningAgent

      You should only have to overwrite getQValue
      and update.  All other QLearningAgent functions
      should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)
    self.weights = util.Counter()

    """
    ************************************************************************
    ************************************************************************
    Aplicaremos epsilon-greedy al proyecto. El Algoritmo se basa en la exploracion 
    y "explotacion"
    Sea Q(S,A) nuestra tabla, Q es el es par de estado(S) accion(A),
    La tabla contiene N estados y M acciones
    (sea e Epsilon)
    En la selección de acción e-greedy, nuestro agente utiliza tanto la explotación (1-e) para aprovechar e
    l conocimiento previo como la exploración (e) para buscar nuevas opciones
    Con una pequeña probabilidad de e, elegimos explorar, es decir,  no explotar lo que 
    hemos aprendido hasta ahora. En este caso, la acción se selecciona aleatoriamente, independientemente de 
    las estimaciones del valor de la acción.
    
    def select-action (Q,s_current_state,e)
      n = rand(1 y 0)
      if n < e then
        A --> Random action from the action space
      else
        A --> MaxQ(S,A)
      end
      return A.
    ************************************************************************
    ************************************************************************
    """

  def getWeights(self):
    return self.weights

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    weights = self.getWeights()
    features = self.featExtractor.getFeatures(state, action)
    q = weights * features
    return q
    util.raiseNotDefined()

  def update(self, state, action, nextState, reward):
    """
        Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    #cogemos los atributos necesarios 
    weights = self.getWeights()
    features = self.featExtractor.getFeatures(state, action)
    actions = self.getLegalActions(nextState)
    maxQ = float("-inf")

    #maxQ de cada action
    for action2 in actions:
      q = self.getQValue(nextState, action2)
      maxQ = max(maxQ, q)
    
    if maxQ == float("-inf"):
      maxQ = 0

    for feature in features:
      #Rt+1 + gammaMaxQ(St+1, a) - Q(St,At)
      difference = (reward + self.discount*maxQ) - self.getQValue(state, action)
      weights[feature] = weights[feature] + self.alpha * difference 

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      return true
