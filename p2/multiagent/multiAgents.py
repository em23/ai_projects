# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        score = 0
        
        # Increasing score if pacman just ate food 
        oldFood_count = currentGameState.getNumFood()
        newFood_count = successorGameState.getNumFood()
        score += (oldFood_count - newFood_count) * 10

        # Decreasing the score depending on how far from the food we go 
        closest = float('inf') if newFood_count else 0
        for xy2 in newFood.asList():   
            value = manhattanHeuristic(newPos, xy2)
            if value < closest: closest = value      
        score -= closest  

        # Increasing the score depending on how far the ghosts are 
        for xy2 in successorGameState.getGhostPositions():   
            score += manhattanHeuristic(newPos, xy2)

        return successorGameState.getScore() + score;


# def mazeDistance(point1, point2, gameState):
#     """
#     Returns the maze distance between any two points, using the search functions
#     you have already built. The gameState can be any game state -- Pacman's
#     position in that state is ignored.

#     Example usage: mazeDistance( (2,4), (5,6), gameState)

#     This might be a useful helper function for your ApproximateSearchAgent.
#     """
#     x1, y1 = point1
#     x2, y2 = point2
#     walls = gameState.getWalls()
#     assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
#     assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
#     prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
#     return len(search.bfs(prob))

def manhattanHeuristic(point1, point2):
    "The Manhattan distance"
    xy1 = point1
    xy2 = point2
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

# def euclideanHeuristic(point1, point2):
#     "The Euclidean distance"
#     xy1 = point1
#     xy2 = point2
#     return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        v, a = self.value(gameState)
        return a
        
        util.raiseNotDefined()

    def value(self, state, agentIndex=0, current_depth=0):
        if state.isWin() or state.isLose(): return self.evaluationFunction(state), 'Stop'
        if current_depth == self.depth * state.getNumAgents(): return self.evaluationFunction(state), 'Stop'
        if agentIndex == 0: return self.max_value(state, agentIndex, current_depth + 1)
        if agentIndex != 0: return self.min_value(state, agentIndex, current_depth + 1)

    def max_value(self, state, agentIndex, current_depth):
        new_agentIndex = (agentIndex + 1) % state.getNumAgents()
        v, a = float('-inf'), None
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            max_v = self.value(successor, new_agentIndex, current_depth)[0]
            if v < max_v:
                v, a = max_v, action
        return v, a

    def min_value(self, state, agentIndex, current_depth):
        new_agentIndex = (agentIndex + 1) % state.getNumAgents()
        v, a = float('inf'), None
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            min_v = self.value(successor, new_agentIndex, current_depth)[0]
            if v > min_v:
                v, a = min_v, action
        return v, a

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        v, a = self.value(gameState)
        return a 

        util.raiseNotDefined()

    def value(self, state, agentIndex=0, current_depth=0, best_max=float('-inf'), best_min=float('inf')):
        if state.isWin() or state.isLose(): return self.evaluationFunction(state), 'Stop'
        if current_depth == self.depth * state.getNumAgents(): return self.evaluationFunction(state), 'Stop'
        if agentIndex == 0: return self.max_value(state, agentIndex, current_depth + 1, best_max, best_min)
        if agentIndex != 0: return self.min_value(state, agentIndex, current_depth + 1, best_max, best_min)

    def max_value(self, state, agentIndex, current_depth, best_max, best_min):
        new_agentIndex = (agentIndex + 1) % state.getNumAgents()
        v, a = float('-inf'), None
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            max_v = self.value(successor, new_agentIndex, current_depth, best_max, best_min)[0]
            if max_v > best_min: return max_v, action
            if max_v > best_max: best_max = max_v
            if v < max_v: v, a = max_v, action
        return v, a

    def min_value(self, state, agentIndex, current_depth, best_max, best_min):
        new_agentIndex = (agentIndex + 1) % state.getNumAgents()
        v, a = float('inf'), None
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            min_v = self.value(successor, new_agentIndex, current_depth, best_max, best_min)[0]
            if min_v < best_max: return min_v, action 
            if min_v < best_min: best_min = min_v
            if v > min_v: v, a = min_v, action
        return v, a

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

