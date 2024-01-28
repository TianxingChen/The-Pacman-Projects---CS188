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
import queue

from util import manhattanDistance
from game import Directions
import random, util
from game import Agent


class ReflexAgent(Agent):

    def getAction(self, gameState):  # 通过评估获得下一步的action， 返回下一步的location
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()  # determine which way is able to go

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]  # get all the action's score
        bestScore = max(scores)  # get best score
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]  # get best action
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalMoves[chosenIndex]

    def bfs_findFood(self, pacman_x, pacman_y, newFood):
        Q = queue.Queue()
        walked, dx, dy, Width, Height = [], [1, 0, -1, 0], [0, 1, 0, -1], newFood.width, newFood.height  # four direction
        Q.put([pacman_x, pacman_y, 0])  # start point
        walked.append((pacman_x, pacman_y))  # record that the start point has reached
        while not Q.empty():  # when Q is not empty
            status = Q.get()
            x, y, step = status
            if newFood[x][y]:
                return step
            for i in range(0, 4):
                px, py = x + dx[i], y + dy[i]
                if (px, py) in walked or px < 0 or py < 0 or px == Width or py == Height:  # reached or overflow
                    continue
                walked.append((px, py))
                Q.put([px, py, step + 1])
        return -1  # represent no food left

    def evaluationFunction(self, currentGameState, action):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)  # Get current Pacman location
        newPos = successorGameState.getPacmanPosition()  # Pacman position
        newFood = successorGameState.getFood()  # Food position
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPos = currentGameState.getGhostPositions()  # Get ghost position

        pacman_x, pacman_y = newPos  # new Pacman position
        # find the closest ghost, using Manhattan distance
        closest_ghost = min([util.manhattanDistance(newPos, ghost) for ghost in ghostPos])
        parameter_ghost = 10  # set for ghost score
        parameter_food = 11  # set for food score
        ghost_dis = 2

        # Calculate ghostScore
        if closest_ghost == 0:
            return float("-inf")

        if closest_ghost and closest_ghost < ghost_dis:
            ghostScore = -parameter_ghost / closest_ghost
        else:
            ghostScore = 0  # set ghostScore -> 0, when ghost hard to influence the Pacman

        # Get foodScore
        if closest_ghost >= ghost_dis:
            # find the closest food
            # closest_food = min([(abs(pos[0] - pacman_x)+abs(pos[1] - pacman_y)) for pos in foodPos])
            # Bfs:
            closest_food = self.bfs_findFood(pacman_x, pacman_y, newFood)  # !!!
            if closest_food == -1:
                foodScore = 0  # fail to find
            else:
                foodScore = parameter_food / closest_food
        else:
            foodScore = 0  # set the score -> 0

        return successorGameState.getScore() + foodScore + ghostScore  # Return evaluated score


def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def minimaxFunc(self, agentIndex, gameState, now_depth):
        # print(self.depth, now_depth)
        if now_depth >= self.depth * 2 or gameState.isWin() or gameState.isLose():  # mult 2
            return self.evaluationFunction(gameState)
        if not agentIndex:  # Pacman
            MAX = float("-inf")
            pacmanAction = gameState.getLegalActions(agentIndex)
            for action in pacmanAction:
                successor_state = gameState.generateSuccessor(agentIndex, action)  # generate new state
                res = self.minimaxFunc(1, successor_state, now_depth + 1)  # recurse
                MAX = max(MAX, res)  # renew MAX value
            return MAX
        else:  # Ghost
            MIN = float("inf")
            ghostAction = gameState.getLegalActions(agentIndex)
            for action in ghostAction:  # Recurse
                successor_state = gameState.generateSuccessor(agentIndex, action)  # generate new state
                if agentIndex >= gameState.getNumAgents() - 1:  # nowAgent is the final ghost
                    MIN = min(MIN, self.minimaxFunc(0, successor_state, now_depth + 1))  # renew MIN value
                else:  # still ghost
                    MIN = min(MIN, self.minimaxFunc(agentIndex + 1, successor_state, now_depth))
            return MIN

    def getAction(self, gameState):
        # util.raiseNotDefined()
        max_value = float("-inf")
        best_expect_action = []
        for action in gameState.getLegalPacmanActions():
            successor_state = gameState.generateSuccessor(0, action)
            res = self.minimaxFunc(0, successor_state, 0)
            if res > max_value:
                max_value = res
                best_expect_action.clear()
                best_expect_action.append(action)
            elif res == max_value:
                best_expect_action.append(action)
        return random.choice(best_expect_action)


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        max_value = float("-inf")
        alpha = float("-inf")  # set alpha and beta into infinite number
        beta = float("inf")
        best_expect_action = []
        for action in gameState.getLegalPacmanActions():
            if action == "Stop":
                continue
            successor_state = gameState.generateSuccessor(0, action)
            res = self.AlphaBeta_minimaxFunc(0, successor_state, 0, alpha, beta)
            if res > max_value:
                max_value = res
                best_expect_action.clear()
                best_expect_action.append(action)
            elif res == max_value:
                best_expect_action.append(action)
            if max_value > res:
                alpha = max_value
        return random.choice(best_expect_action)  # rand

    def AlphaBeta_minimaxFunc(self, agentIndex, gameState, now_depth, alpha, beta):
        if now_depth >= self.depth or gameState.isWin() or gameState.isLose():  # mult 2
            # return self.evaluatationFunc(gameState)
            return betterEvaluationFunction(gameState)
        if agentIndex == 0:  # Pacman
            MAX = float("-inf")
            pacmanAction = gameState.getLegalActions(agentIndex)
            for action in pacmanAction:
                successor_state = gameState.generateSuccessor(agentIndex, action)  # generate new state
                res = self.AlphaBeta_minimaxFunc(1, successor_state, now_depth + 1, alpha, beta)  # recurse
                if res is not None:
                    MAX = max(MAX, res)  # renew MAX value
                if MAX > beta:  # cut
                    return MAX
                if MAX > alpha:
                    alpha = MAX
            return MAX

        else:  # Ghost
            MIN = float("inf")
            ghostAction = gameState.getLegalActions(agentIndex)
            for action in ghostAction:  # Recurse
                successor_state = gameState.generateSuccessor(agentIndex, action)  # generate new state
                # if agentIndex >= gameState.getNumAgents() - 1:  # nowAgent is the final ghost
                #     MIN = min(MIN, self.AlphaBeta_minimaxFunc(0, successor_state, now_depth + 1, alpha,beta))  # renew MIN value
                # else:  # still ghost
                #     MIN = min(MIN, self.AlphaBeta_minimaxFunc(agentIndex + 1, successor_state, now_depth, alpha, beta))
                res = self.AlphaBeta_minimaxFunc((agentIndex+1) % gameState.getNumAgents(), successor_state, now_depth, alpha, beta)
                if res is not None:
                    MIN = min(MIN, res)
                if MIN < alpha:
                    return MIN
                if MIN < beta:
                    beta = MIN
            return MIN


def betterEvaluationFunction(currentGameState):
    # util.raiseNotDefined()
    currentPos = currentGameState.getPacmanPosition()  # Pacman position
    currentFood = currentGameState.getFood()  # Food position
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    ghostPos = currentGameState.getGhostPositions()  # Get ghost position

    pacman_x, pacman_y = currentPos  # new Pacman position

    # Ghost
    # ghostScore, totalDistance = 0, 0
    # for ghostPos in currentGameState.getGhostPositions():
    #     ghostDistance = util.manhattanDistance(currentPos, ghostScore)
    #     if ghostDistance <= 1:  # if Pacman is going to die, return negative inf
    #         return float("-inf")
    #     totalDistance += ghostDistance
    # ghostScore = -1 / float(totalDistance)  # negative

    ghostScore, totalDistance = 0, 0
    for i in range(len(currentGameState.getGhostPositions())):
        ghost = currentGameState.getGhostPositions()[i]
        dis = util.manhattanDistance(ghost, currentPos)
        if not currentScaredTimes[i] and dis <= 1:
            return float("-inf")
        totalDistance += dis
    ghostScore = -1 / float(totalDistance)  # negative

    # Food
    closest_food = bfs_findFood(pacman_x, pacman_y, currentFood)
    foodScore = 1 / float(closest_food)

    # Capsule
    CapsuleScore = float("-inf")  # extreme small value
    CapsulesPos = currentGameState.getCapsules()
    for capsule in CapsulesPos:
        CapsuleScore = max(CapsuleScore, 1.5 / float(util.manhattanDistance(capsule, currentPos)))

    foodScore = max(foodScore, CapsuleScore)

    return currentGameState.getScore() + foodScore + ghostScore


def bfs_findFood(pacman_x, pacman_y, newFood):
    Q = queue.Queue()
    walked, dx, dy, Width, Height = [], [1, 0, -1, 0], [0, 1, 0,-1], newFood.width, newFood.height  # four direction
    Q.put([pacman_x, pacman_y, 0])  # start point
    walked.append((pacman_x, pacman_y))  # record that the start point has reached
    while not Q.empty():  # when Q is not empty
        status = Q.get()
        x, y, step = status
        if newFood[x][y]:
            return step
        for i in range(0, 4):
            px, py = x + dx[i], y + dy[i]
            if (px, py) in walked or px < 0 or py < 0 or px == Width or py == Height:  # reached or overflow
                continue
            walked.append((px, py))
            Q.put([px, py, step + 1])
    return -1  # represent no food left


# Abbreviation
better = betterEvaluationFunction
