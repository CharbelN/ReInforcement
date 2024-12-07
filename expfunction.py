
from learningAgents import ReinforcementAgent
from backend import ReplayMemory
import csv
import backend
#import gridworld


import random,util,math
import numpy as np

class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter()
        self.visitCounts = util.Counter()  # To track visit counts for states
        self.cumulative_reward = 0  # Cumulative reward for the current episode
        self.filename = "exploration_K2.csv"  # File to save performance data
        self.episode_count=0
        # Initialize the CSV file and write the header
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["episode", "cumulative_reward", "average_q_value", "epsilon"])
            writer.writeheader()

    def getQValue(self, state, action):
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        return max([self.explorationFunction(state, action) for action in legalActions])

    def computeActionFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        best_action = None
        best_value = float('-inf')

        for action in legalActions:
            value = self.explorationFunction(state, action)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward: float,done:bool):
        #print(self.episode_count)
        self.cumulative_reward += reward
        #print(self.cumulative_reward)
        self.visitCounts[(state, action)] += 1  # Increment visit count
        currentQValue = self.getQValue(state, action)
        nextValue = self.computeValueFromQValues(nextState)
        updatedQValue = ((1 - self.alpha) * currentQValue) + (self.alpha * (reward + self.discount * nextValue))
        self.qValues[(state, action)] = updatedQValue
        if done:
            self.episode_count += 1
            average_q_value = np.mean(list(self.qValues.values()))

            # Append data to the CSV file
            with open(self.filename, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["episode", "cumulative_reward", "average_q_value", "epsilon"])
                writer.writerow({
                    "episode": self.episode_count,
                    "cumulative_reward": self.cumulative_reward,
                    "average_q_value": average_q_value,
                    "epsilon": self.epsilon,
                })

            # Reset cumulative reward for the next episode
            self.cumulative_reward = 0






    def explorationFunction(self, state, action,k=3):
        """
        Exploration function f(u, n) = u + k / sqrt(n)
        where:
        - u is the Q-value (self.getQValue(state, action))
        - n is the visit count (self.visitCounts[(state, action)])
        """
        u = self.getQValue(state, action)
        n = self.visitCounts[(state, action)]
       
        #print(u)
        if n == 0:  # Handle the case for unvisited actions
           
            return u + k  # Assign maximum optimism for unvisited actions
        #print(u+k/math.sqrt(n))
        return u + k / math.sqrt(n)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
