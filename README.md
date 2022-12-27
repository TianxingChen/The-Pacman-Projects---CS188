# The-Packman-Projects---CS188

One of the CS188's projects, based on MiniMax-Searching Agent
Programming Language: **Python**

## List:
* Reflex Agent
* Evaluate function
* Minimax
* Alpha-Beta
* Better-evaluateFunction

[Click here to view more](https://www.marioctx.com/2022/11/11/The-Pac-Man-Projects)

---
title: The Pac-Man Projects
date: 2022-11-11 19:55:36
tags: 
- AI
- Projects
categories: Projects
top:

---

# Pac-man —— 对抗搜索

对**Pac-man**游戏编写基于minimax搜索的智能体，<u>CS188</u>经典实验：[Project Link：CS188-Pacman](http://ai.berkeley.edu/project_overview.html)

Github Link:[Click here](https://github.com/MarioTX/The-Pacman-Projects---CS188)

My code download: [Click here](https://pacman-cs188-1312128486.cos.ap-nanjing.myqcloud.com/The%20Pacman%20Project.zip)

![](https://pacman-cs188-1312128486.cos.ap-nanjing.myqcloud.com/WechatIMG38.png)

# Problem one: Reflex智能体

一个粗略的**评估函数**，综合了被Ghost吃到的风险以及吃豆得分，简单评估最优的去向。

## Function: getAction

评估所有可行行为，并且选择一个进行操作。函数定义等已经完备，不必要改变

```python
def getAction(self, gameState):  # assess action and get next position
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()  # determine which way is able to go

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]  # get all the action's score
        bestScore = max(scores)  # get best score
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]  # get best action
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        
        return legalMoves[chosenIndex]
```

## evaluationFunction

评估分数的函数，基于Manhattan distance

### 算法实现思路

首先保证能赢得游戏（Win），所以如果即将要被吃到的话，我们选择优先躲避Ghost，否则的话我们选择得分期望最大的方向去跑

1. 设定参数（**parameter_ghost** and **parameter_food**），定义变量

   ```python
   pacman_x, pacman_y = newPos  # new Pacman position
   # find the closest ghost, using Manhattan distance
   closest_ghost = min([util.manhattanDistance(newPos, ghost) for ghost in ghostPos])
   parameter_ghost = 11  # set for ghost sycore
   parameter_food = 10  # set for food score
   ```

2. 计算ghostScore

   ```python
   # Calculate ghostScore
   if closest_ghost and closest_ghost < 2:
   	ghostScore = -parameter_ghost/closest_ghost
   else:
   	ghostScore = 0  # set ghostScore -> 0, when ghost hard to influence the Pacman
   ```

3. 计算foodScore

   如果最近的（威胁最大）的ghost距离很近，那么我们本着首先要赢得游戏的原则，需要先选择躲避，否则找到最近的food去吃，当然寻找最近食物可以用**暴力**和**bfs**写:

   #### 暴力——O(Width*Height)

   首先，要找出所有的剩余食物，坐标 **(i, j)**

   ```python
   foodPos = []  # Array -> all food's position
           Width, Height = newFood.width, newFood.height
           for i in range(Width):  # [0, Width)
               for j in range(Height):  # [0, Height)
                   if newFood[i][j]:  # Which means that (i, j) exist food
                       foodPos.append((i, j))  # Put the food into the pos
   ```

   计算得分

   ```python
   # Get foodScore
   if closest_ghost >= 2 and len(foodPos):
     # find the closest food
     closest_food = min([(abs(pos[0] - pacman_x)+abs(pos[1] - pacman_y)) for pos in foodPos]) 
     foodScore = parameter_food/closest_food
   else:
     foodScore = 0  # set the score -> 0
   ```

   #### BFS——O(1)~O(Width*Height) 

   ```python
   def bfs_findFood(pacman_x, pacman_y, newFood):
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
               Q.put([px, py, step+1])
       return -1  # represent no food left
   ```

   计算得分

   ```python
   # Get foodScore
   if closest_ghost >= 2:
     # find the closest food
     closest_food = self.bfs_findFood(pacman_x, pacman_y, newFood)  # !!!
     if closest_food == -1:
       	foodScore = 0  # fail to find
     else:
      	  foodScore = parameter_food/closest_food
   else:
      foodScore = 0  # set the score -> 0
   ```

4. 综合得分，返回评估

   ```python
   return successorGameState.getScore() + foodScore + ghostScore  # Return evaluated score
   ```

## 运行评估

Reflex智能体运行评估：（10次）

```C++
python pacman.py -p ReflexAgent -l openClassic -n 10 -q
```

运行结果：

![](https://pacman-cs188-1312128486.cos.ap-nanjing.myqcloud.com/WechatIMG40.png)

## 个人评估

主要是对于躲避距离的参数调整，也就是在**Problem one**中，没有障碍以及只有一个ghost，所以将极限距离设置为2也能保证必胜，但是如果有了各种墙的约束以及多ghost追逐，显然2的设定并不合理，所以在最后整合的代码中有以下**权衡**：

* 提高距离的约束设定
* 随着存活难度的增加适当提高躲避的比例，也就是增大`parameter_ghost/parameter_food`。

# Problem two: Minimax

通过Minimax实现**对抗性搜索智能体**

根据写好的代码模版，可以简单推知：**agentIndex用0代表Pacman，用>=1代表Ghost**。至于`getLegalActions`等需要分类讨论的方法，代码模版已经集成好了，指需要通过接口传入**agentIndex**的参数就可以自动分类。

## minimaxAgent

通过**depth**限制递归层数（相当于边界），根据题目要求，我们将一个max层和min层统一为一个层

```python
def minimaxFunc(self, agentIndex, gameState, now_depth):
  if now_depth >= self.depth*2 or gameState.isWin() or gameState.isLose():  # mult 2
    return self.evaluationFunction(gameState)
  if agentIndex == 0:  # Pacman
    MAX = float("-inf")
    pacmanAction = gameState.getLegalActions(agentIndex)
    for action in pacmanAction:
      successor_state = gameState.generateSuccessor(agentIndex, action)  # generate new state
      res = self.minimaxAgent(0, successor_state, now_depth)  # recurse
      MAX = max(MAX, res)  # renew MAX value
      return MAX
  else:  # Ghost
    MIN = float("inf")
    ghostAction = gameState.getLegalActions(agentIndex)
    for action in ghostAction:  # recurse
      successor_state = gameState.generateSuccessor(agentIndex, action)  # generate new state
      if agentIndex == gameState.getNumAgents()-1:  # nowAgent is the final ghost
        MIN = min(MIN, self.minimaxAgent(0, successor_state, now_depth+1))  # renew MIN value
      else:  # still ghost
        MIN = min(MIN, self.minimaxAgent(1, successor_state, now_depth))
        return MIN
```

搞定了**minimaxFunc**，我们写一个整合函数**getAction**来调用并且返回最佳评估方向：

```python
def getAction(self, gameState):  # return best expect action
  max_value = float("-inf")
  best_expect_action = "Stop"  # default action
  for action in gameState.getLegalPacmanActions():
    successor_state = gameState.generateSuccessor(0, action)
    res = self.minimaxFunc(0, successor_state, 0)
    if res is not None and res > max_value:
      max_value = res
      best_expect_action = action
return best_expect_action
```

## 运行评估

测试指令：

```
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3 -q -n 100
```

运行结果：

![](https://pacman-cs188-1312128486.cos.ap-nanjing.myqcloud.com/1668612158268.jpg)

# Problem three: Alpha-Beta剪枝

通过剪枝技巧加速**minimax**对抗搜索，具备完备性和最优性.

Alpha表示下界，Beta表示上界，具体的算法实现可以参考：[From oiwiki](https://oi-wiki.org//search/alpha-beta/)

>在局面确定的双人对弈里，常进行对抗搜索，构建一棵每个节点都为一个确定状态的搜索树。奇数层为己方先手，偶数层为对方先手。搜索树上每个叶子节点都会被赋予一个估值，估值越大代表我方赢面越大。我方追求更大的赢面，而对方会设法降低我方的赢面，体现在搜索树上就是，奇数层节点（我方节点）总是会选择赢面最大的子节点状态，而偶数层（对方节点）总是会选择我方赢面最小的的子节点状态。

## Alpha-Beta优化的minimaxFunc

在原有的**minimaxFunc**中通过传递alpha和beta进行简单的剪枝：

#### AlphaBeta_minimaxFunc

```python
def AlphaBeta_minimaxFunc(self, agentIndex, gameState, now_depth, alpha, beta):
    if now_depth >= self.depth * 2 or gameState.isWin() or gameState.isLose():  # mult 2
        return self.evaluationFunction(gameState)
    if not agentIndex:  # Pacman
        MAX = float("-inf")
        pacmanAction = gameState.getLegalActions(agentIndex)
        for action in pacmanAction:
            successor_state = gameState.generateSuccessor(agentIndex, action)  # generate new state
            res = self.AlphaBeta_minimaxFunc(1, successor_state, now_depth+1, alpha, beta)  # recurse
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
            if agentIndex >= gameState.getNumAgents()-1:  # nowAgent is the final ghost
                MIN = min(MIN, self.AlphaBeta_minimaxFunc(0, successor_state, now_depth+1, alpha, beta))  # renew MIN value
            else:  # still ghost
                MIN = min(MIN, self.AlphaBeta_minimaxFunc(agentIndex+1, successor_state, now_depth, alpha, beta))
            if MIN < alpha:
                return MIN
            if MIN < beta:
                beta = MIN
        return MIN
```

#### getAction

```python
def getAction(self, gameState):
    max_value = float("-inf")
    alpha = float("-inf")  # set alpha and beta into infinite number
    beta = float("inf")
    best_expect_action = []
    for action in gameState.getLegalPacmanActions():
        if action == "Stop":  # Pacman have to move !!!
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
```

## 优化

如果evaluateFunc就如上面那样写，我们会发现可爱的吃豆人基本赢不了，因为对于死亡状态的**惩罚**太少了，我们当然应该先保证游戏可以胜利，所以我们只要在**getAction**中加上一行代码就可以了：

```python
if closest_ghost == 0:  # When Pacman is going to die, return negative inf
    return float("-inf")
```

以及为了驱动**Pacman**采取行动，我们**ban**掉其不动的**action**，查阅**game.py**会发现停止的指令为`'Stop'`

```python
if action == "Stop":  # Pacman have to move !!!
            continue
```



## 运行评估

执行以下语句（基于**smallClassic**布局）：

* 只有一个**ghost**，搜索深度为4，运行100回合：

```
python pacman.py -p AlphaBetaAgent -a depth=4 -l smallClassic -k 1 -q -n 100
```

![](https://pacman-cs188-1312128486.cos.ap-nanjing.myqcloud.com/WechatIMG1652.png)

* 有两个**ghost**，搜索深度为4，运行100回合：

```
python pacman.py -p AlphaBetaAgent -a depth=4 -l smallClassic -q -n 100
```

![](https://pacman-cs188-1312128486.cos.ap-nanjing.myqcloud.com/WechatIMG33.png)

### 结果评估

留意到，如果只有一个**ghost**，胜率将会较高（84%），而且得分也较高（847.28）；但是如果我们将**ghost**的数量提高，胜率就会大幅度降低（19%），得分也波动很大（150.13）。综合算法较为简单的因素，总体可以判定算法有效。

# Problem four: betterEvaluationFunction

根据**Problem three**的实验结果：

* Pacman经常死亡，因为在**smallClassic**下ghost的数量>1，但是我们对于**ghostScore**的评估只取威胁最大的一个（最近），所以可能会导致顾此失彼。

* 发现模型很难突破<u>1000</u>分，因为想要获取高分，胜利并不够，还要通过吃胶囊加分，简单观察也能发现**evaluateFunc**中的**newScaredTimes**，也就是关于ghost的恐惧时间并没有用上；

所以在此问中进行改善。

## Evaluated value for ghost

针对第一个问题，我们可以利用分式的性质，对所有ghost进行分式**求和**，或者只对所有**manhattan distance**小于某个边界值的ghost进行加权评估，这里我们选择比较简单的第一种处理方式。

```python
ghostScore, totalDistance = 0, 0
    for ghostPos in currentGameState.getGhostPositions():
        ghostDistance = util.manhattanDistance(currentPos, ghostScore)
        if ghostDistance <= 1:  # if Pacman is going to die, return negative inf
            return float("-inf")
        totalDistance += ghostDistance
    ghostScore = -1 / float(totalDistance)  # negative!
```

为了防止出现除0的现象，以及为了保证游戏的胜利，当**ghostDistance**小于等于1的时候，我们就返回一个 **-inf**，以保证尽量不败：

```python
if ghostDistance <= 1:
	return float("-inf")
```



## Evaluated value for food

利用**BFS**求出最近的food

```python
closest_food = bfs_findFood(pacman_x, pacman_y, currentFood)
foodScore = 1 / float(closest_food)
```

### evaluated value for capsule

**!!! Problem four 的精髓**，将capsule和food看成是同类的元素，同时capsule的权重设置的更高一点

```python
CapsuleScore = float("-inf")  # extreme small value
CapsulesPos = currentGameState.getCapsules()
for capsule in CapsulesPos:
    CapsuleScore = max(CapsuleScore, 1.5 / float(util.manhattanDistance(capsule, currentPos)))
```

**将food和capsule取一个极大值并且作为总体得分**

```python
foodScore = max(foodScore, CapsuleScore)
```



## Debug

按上面那样写，看起来非常的有道理，但是我们忽略了一个问题：

在对于**ghost**的评估中，我们写了个`if ghostDistance <= 1`即返回极差评估的判断，看起来没有问题，但是在这个**betterEvaluationFunction**中，我们开始考虑吃胶囊，也就是当ghost靠近的时候，我们不一定要逃离，反而可以通过吃胶囊的方式**反杀**。

我的判断条件比较简易：只有`ghostDistance <= 1`和`currentScaredTimes == 0`同时成立时才返回-inf，这样保证了拿到了胶囊会选择反杀决策，如果有生命危险而且没有胶囊就速run：

```python
ghostScore, totalDistance = 0, 0
for i in range(len(currentGameState.getGhostPositions())):
    ghost = currentGameState.getGhostPositions()[i]
    dis = util.manhattanDistance(ghost, currentPos)
    if not currentScaredTimes[i] and dis <= 1:  # !!!
        return float("-inf")
    totalDistance += dis
ghostScore = -1 / float(totalDistance)  # negative
```



## 运行评估

基于**smallClassic**布局，记得要把**AlphaBeta_minimaxFunc**中的边界返回值函数改成`betterEvaluationFunction`

* 只有一个**ghost**，搜索深度为4，运行10回合：

```
python pacman.py -p AlphaBetaAgent -a depth=4 -l smallClassic -k 1 -q -n 10
```

![](https://pacman-cs188-1312128486.cos.ap-nanjing.myqcloud.com/WechatIMG1660.png)

* 有两个**ghost**，搜索深度为4，运行10回合：

```
python pacman.py -p AlphaBetaAgent -a depth=4 -l smallClassic -q -n 100
```

![](https://pacman-cs188-1312128486.cos.ap-nanjing.myqcloud.com/WechatIMG37.png)

#### 结果评估

胜率和得分都有较高的提升！加入对胶囊的考虑确实对结果有这积极影响。

# 实验总结

还是有很多地方可以优化，由于时间限制很多可评估方向并没有作处理，比如对于**反杀**的操作策略还比较baby，不过最后的智能体也能在single Ghost的情况下拿到**1k+Score average**，总体也是不错的实验结果。

<div style="position: relative; width: 100%; height: 0; padding-bottom: 75%;"><iframe src="https://pacman-cs188-1312128486.cos.ap-nanjing.myqcloud.com/11%E6%9C%8816%E6%97%A5.mp4" scrolling="no" border="0" 
frameborder="no" framespacing="0" allowfullscreen="true" style="position: absolute; width: 100%; 
height: 100%; left: 0; top: 0;"> </iframe></div> 


# API

```python
gameState.getLegalActions(agentIndex):
Returns a list of legal actions for an agent
agentIndex=0 means Pacman, ghosts are >= 1

gameState.generateSuccessor(agentIndex, action):
Returns the successor game state after an agent takes an action

gameState.getNumAgents():
Returns the total number of agents in the game

gameState.isWin():
Returns whether or not the game state is a winning state

gameState.isLose():
Returns whether or not the game state is a losing state
```

# Commands

* 命令 `-g DirectionalGhost` 使游戏中的幽灵更聪明和有方向性；
* 你也可以通过命令 `-n` 来让游戏运行多次；
* 使用命令 `-q ` 关闭图形化界面使游戏更快运行；
* 用 `-–frameTime 0` 加速游戏画面。

**智能体**的可视化： 

```
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=4
```

## Problem one:

```c++
python pacman.py    // 通过键盘方向键控制
python pacman.py -p ReflexAgent -l testClassic    // 在默认布局测试Reflex智能体
python pacman.py -p ReflexAgent -l testClassic -g DirectionalGhost    // -g DirectionalGhost 使游戏中的幽灵更聪明和有方向性
python pacman.py -p ReflexAgent -l testClassic -g DirectionalGhost -q -n 10    // -q 关闭图形化界面使游戏更快运行  -n 让游戏运行多次，这段命令是运行了十次
python pacman.py --frameTime 0 -p ReflexAgent -k 2    // 设置2个幽灵，同时提升加速显示
python pacman.py --frameTime 0 -p ReflexAgent -k 1    // 设置1个幽灵，同时提升加速显示

python pacman.py -p ReflexAgent -l openClassic    // 在openClassic布局测试Reflex智能体
python pacman.py -p ReflexAgent -l openClassic -q -n 15
python pacman.py -p ReflexAgent -l openClassic -g DirectionalGhost
python pacman.py -p ReflexAgent -l openClassic -g DirectionalGhost -q -n 10
```

## Problem two:

```c++
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=3 -q -n 10
python pacman.py -p MinimaxAgent -l openClassic -a depth=3 -g DirectionalGhost
python pacman.py -p MinimaxAgent -l mediumClassic -a depth=3 -g DirectionalGhost
python pacman.py -p MinimaxAgent -l smallClassic -a depth=3 -g DirectionalGhost
```

## Problem three:

```C++
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic -k 1 -q -n 10
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic -k 1 -g DirectionalGhost -q -n 10
python pacman.py -p AlphaBetaAgent -a depth=2 -l smallClassic -g DirectionalGhost
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic -g DirectionalGhost
```

