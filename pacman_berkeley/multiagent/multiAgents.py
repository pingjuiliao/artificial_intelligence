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
        """
        print "#" * 30
        print newPos
        print newFood
        for state in newGhostStates :
            print dir(state)
        print newScaredTimes
        print "#" * 30
        print "/" * 30
        """

        manhattan = lambda x0, y0, x1, y1 : abs(x0 - x1) + abs(y0 - y1)
        food_remain = newFood.count()
        distance = newFood.width + newFood.height
        for x in range(1, newFood.width) :
            for y in range(1, newFood.height) :
                if x == newPos[0] and y == newPos[1] :
                    food_remain -= 1
                    continue
                if newFood[x][y] :
                    d = manhattan(x, y, newPos[0], newPos[1])
                    distance = min(d, distance)
        # print distance

        standard = newFood.width + newFood.height
        score  = 0
        score -= distance
        score -= food_remain * standard
        for ghost in newGhostStates :
            if ghost.scaredTimer == 0 :
                gx, gy = ghost.getPosition()
                peripheral = [(gx, gy), (gx+1, gy), (gx-1, gy), (gx, gy+1), (gx, gy-1)]
                if newPos in peripheral :
                    score -= standard * standard
        # print "score is >>> ", score
        return score

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

        agent_index_pacman = 0 ## 0 for pacman, n >= 1 for ghosts


        def minimax(state, agentIdx, depth) :

            scores = []
            next_agentIdx = (agentIdx + 1) % state.getNumAgents()
            next_depth = depth - 1 if next_agentIdx == 0 else depth
            for action in state.getLegalActions(agentIdx) :
                succ_state = state.generateSuccessor(agentIdx, action)
                if next_depth <= 0 or \
                    len(succ_state.getLegalActions(next_agentIdx)) == 0:
                    scores.append((self.evaluationFunction(succ_state), action))
                else :
                    scores.append((minimax(succ_state, next_agentIdx, next_depth)[0], action))
            if agentIdx >= 1 :
                result = min(scores)
            else :
                result = max(scores)

            return result


        score, act = minimax(gameState, agent_index_pacman, self.depth)
        return act

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta_pruning(state, alpha, beta, agentIdx, depth) :
            scores = []
            next_agentIdx = ( agentIdx + 1 ) % state.getNumAgents()
            next_depth = depth - 1 if next_agentIdx == 0 else depth
            curr_max, curr_min = -float('Inf'), float('Inf')
            for action in state.getLegalActions(agentIdx) :
                #if depth == self.depth :
                #    print "Current alpha, beta are", alpha, beta
                ## get score
                succ_state = state.generateSuccessor(agentIdx, action)
                if len(succ_state.getLegalActions(agentIdx)) == 0 or \
                    next_depth == 0 :
                    score = (self.evaluationFunction(succ_state), action)
                    #print "Pure Score : ", score
                else :
                    next_opt = alpha_beta_pruning(succ_state, alpha, \
                                        beta, next_agentIdx, next_depth)
                    score = (next_opt[0], action)
                scores.append(score)
                print scores
                ## alpha-beta pruning
                if agentIdx == 0 :
                    if score[0] > beta :
                        print "path taken 1, score > beta ==", score, " > ", beta
                        return score
                    alpha = max(score[0], alpha)
                else :
                    if score[0] < alpha :
                        print "path taken , score < alpha ==", score, " < ", alpha
                        return score
                    beta = min(score[0], beta)
            if agentIdx == 0 :
                result = max(scores)
            else :
                result = min(scores)
            return result
            ## update scores
        pacman_idx = 0
        score, act = alpha_beta_pruning(gameState, -float('Inf'), float('Inf'), pacman_idx, self.depth)
        return act

        util.raiseNotDefined()

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
        def expectimax(state, agentIdx, depth) :

            scores = []
            next_agentIdx = ( agentIdx + 1 ) % state.getNumAgents()
            next_depth = depth - 1 if next_agentIdx == 0 else depth
            for action in state.getLegalActions(agentIdx) :
                succ_state = state.generateSuccessor(agentIdx, action)
                if next_depth == 0 or \
                        len(succ_state.getLegalActions(agentIdx)) == 0 :
                    score = self.evaluationFunction(succ_state)
                else :
                    score, _ = expectimax(succ_state, next_agentIdx, next_depth)
                scores.append((score, action))

            if agentIdx == 0 :
                return max(scores)
            else :
                average = sum([tup[0] for tup in scores]) / float(len(scores))
                #print average
                return (average, "None")

        pacman_idx = 0
        score, act = expectimax(gameState, pacman_idx, self.depth)
        return act

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # print dir(currentGameState)
    wallGrid = currentGameState.getWalls().copy()

    pacman = currentGameState.getPacmanState()
    pacman_pos = pacman.getPosition()

    print "=" * 30
    print ">>>this pacman is towarding :", pacman_pos
    # print dir(pacman)
    # print pacman
    paths = path_define_function(wallGrid)
    standard = wallGrid.height + wallGrid.width
    capsules_list = currentGameState.getCapsules()
    foodGrid = currentGameState.getFood()

    ## see capsule as food
    for x, y in capsules_list :
        foodGrid[x][y] = True


    ## manhattan score
    manhattan = lambda src, dst : abs(src[0] - dst[0]) + abs(src[1] - dst[1])
    min_distance = float('Inf')
    if currentGameState.getNumFood() == 0 :
        manhattan_score = 100
    else :
        for x in range(1, foodGrid.width-1) :
            for y in range(1, foodGrid.height-1) :
                if foodGrid[x][y] :
                    food = (x , y)
                    min_distance = min(min_distance, manhattan(food, pacman_pos))

    print "min_distance is ", min_distance
    manhattan_weight = standard * 1
    manhattan_score  = - min_distance
    manhattan_item   = manhattan_weight * manhattan_score
    print "\nmanhattan_weight * manhattan_score == manhattan_item"
    print "{} * {} == {}".format(manhattan_weight, manhattan_score, \
                                    manhattan_item)
    ## bfs search
    bfs_depth_score = bfs_depth(pacman_pos, foodGrid, wallGrid)
    bfs_weight = standard ** 1
    bfs_score  = - bfs_depth_score
    bfs_item   = bfs_weight * bfs_score
    print "\nbfs_weight * bfs_score == bfs_item"
    print bfs_weight, " * ", bfs_score, " == ", bfs_item

    ## ghost avoiding
    surrounds = lambda x ,y : [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    legalMoves = lambda x, y : \
        [ pos for pos in surrounds(x, y) if not wallGrid[pos[0]][pos[1]] ]

    ghost_score = 0
    for next_pos in legalMoves(pacman_pos[0], pacman_pos[1]) :
        for ghost_pos in currentGameState.getGhostPositions() :
            if manhattan(ghost_pos, next_pos) <= 1 :
                ghost_score = -10

    ghost_weight = standard ** 3 if pacman.scaredTimer == 0 else 0
    ghost_item = ghost_weight * ghost_score

    ## win
    win_item = standard ** 5 if currentGameState.isWin() else 0

    ## food remained
    food_remaining = currentGameState.getNumFood()
    food_remaining = food_remaining - 1 \
            if currentGameState.hasFood(pacman_pos[0], pacman_pos[1]) else food_remaining
    food_remain_score = - food_remaining
    food_remain_weight = standard ** 2 if food_remaining != 0 else standard ** 3
    food_remain_item = food_remain_weight * food_remain_score


    ## linear formula
    result = manhattan_item + bfs_item + ghost_item + food_remain_item

    if foodGrid.count() <= 1 :
        result = bfs_item + currentGameState.getScore()

    print ">> result == ", result

    return result
    util.raiseNotDefined()

def path_define_function(walls) :

    ## path algorithm :
    ## defining the following
    # 1. a path with a deadend and an exit definition
    # 2. a path with two exits
    reachable = lambda a,x,y : \
        [a[x+1][y], a[x-1][y], a[x][y+1], a[x][y-1]].count(False)
    legalMoves = lambda x,y : \
        [ e for e in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)] \
                    if not walls[e[0]][e[1]] ]
    visited = walls.copy()
    # test
    """
    print "W" * walls.width
    for y in range(1, walls.height - 1) :
        s = "W"
        for x in range(1, walls.width - 1) :
            if walls[x][y] :
                s += "W"
            else :
                s += str(reachable(walls, x, y))
        s += "W"
        print s
    print "W" * walls.width
    """
    ## legal
    path_map = []
    for startx in range(1, walls.width - 1) :
        for starty in range(1, walls.height - 1) :
            if visited[startx][starty] :
                continue
            ## not visited :
            visited[startx][starty] = True
            if reachable(walls, startx, starty) >= 3 :
                continue
            ## now the available path is only 2 or 1

            fringe = util.Stack()
            fringe.push((startx, starty))
            result = []
            while not fringe.isEmpty() :

                x, y = fringe.pop()
                visited[x][y] = True
                if reachable(walls, x, y) == 1 :
                    result.append((x, y))

                for newx, newy in legalMoves(x, y) :
                    # print "expanding ({}, {})".format(newx, newy)
                    if reachable(walls, newx, newy) >= 3 :
                        result.append((x, y))
                    elif not visited[newx][newy] :
                        fringe.push((newx, newy))

            path_map.append(result)

    debug = False
    if debug :
        print "Path Results >>> "
        for a in path_map:
            print a
        print walls

    ## remove path with one dot
    complicated_path_map = [ path for path in path_map if path[0] != path[1] ]

    if debug :
        print complicated_path_map
        print walls
    # return path_map
    return complicated_path_map



def legalPos(pos, wallgrid) :
    x, y = pos
    if wallgrid[x][y] :
        return False
    if x < 0 or wallgrid.width <= x :
        return False
    if y < 0 or wallgrid.height <= y :
        return False
    return True

def legalNextPos(x, y, wallgrid) :
    arr = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    return [ pos for pos in arr if legalPos((pos[0], pos[1]), wallgrid)]


def bfs_depth(pacman, food, wall) :

    ## init
    NO_FOOD_DEPTH = -1
    visited = []
    fringe = [ pacman ]
    depth = 0

    ## expand
    while fringe != [] :
        for x, y in fringe :
            if food[x][y] :
                return depth
        depth += 1
        visited += fringe
        nextFringe = []
        for pos in fringe :
            nextFringe += [ npos for npos in legalNextPos(pos[0], pos[1], wall) \
                                if npos not in visited ]
        fringe = nextFringe
    return NO_FOOD_DEPTH



# Abbreviation
better = betterEvaluationFunction

