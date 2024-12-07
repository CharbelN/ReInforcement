
from learningAgents import ReinforcementAgent
from backend import ReplayMemory

import backend
#import gridworld


import random,util,math
import numpy as np
import copy

# class QLearningAgent(ReinforcementAgent):
#     """
#       Q-Learning Agent
#       Functions you should fill in:
#         - computeValueFromQValues
#         - computeActionFromQValues
#         - getQValue
#         - getAction
#         - update
#       Instance variables you have access to
#         - self.epsilon (exploration prob)
#         - self.alpha (learning rate)
#         - self.discount (discount rate)
#       Functions you should use
#         - self.getLegalActions(state)
#           which returns legal actions for a state
#     """
#     def __init__(self, **args):
#         "You can initialize Q-values here..."
#         ReinforcementAgent.__init__(self, **args)

#         "*** YOUR CODE HERE ***"
#         self.qValues = util.Counter()  # A Counter is a dict with default 0
#         print("THE INITIAL EPSILON VALUE" , self.epsilon)

#     def getQValue(self, state, action):
#         """
#           Returns Q(state,action)
#           Should return 0.0 if we have never seen a state
#           or the Q node value otherwise
#         """
#         "*** YOUR CODE HERE ***"
#         return self.qValues[(state, action)]

#     def computeValueFromQValues(self, state):
#         """
#           Returns max_action Q(state,action)
#           where the max is over legal actions.  Note that if
#           there are no legal actions, which is the case at the
#           terminal state, you should return a value of 0.0.
#         """
#         "*** YOUR CODE HERE ***"
#         legalActions = self.getLegalActions(state)
#         if not legalActions:
#             return 0.0

#         # Return the maximum Q-value over all possible actions
#         return max([self.getQValue(state, action) for action in legalActions])

#     def computeActionFromQValues(self, state):
#         """
#           Compute the best action to take in a state.  Note that if there
#           are no legal actions, which is the case at the terminal state,
#           you should return None.
#         """
#         "*** YOUR CODE HERE ***"
#         legalActions = self.getLegalActions(state)
#         if not legalActions:
#             return None

#         # Compute the best action
#         best_action = None
#         best_value = float('-inf')

#         for action in legalActions:
#             q_value = self.getQValue(state, action)
#             if q_value > best_value:
#                 best_value = q_value
#                 best_action = action

#         return best_action

#     def getAction(self, state):
#         """
#           Compute the action to take in the current state.  With
#           probability self.epsilon, we should take a random action and
#           take the best policy action otherwise.  Note that if there are
#           no legal actions, which is the case at the terminal state, you
#           should choose None as the action.
#           HINT: You might want to use util.flipCoin(prob)
#           HINT: To pick randomly from a list, use random.choice(list)
#         """
#         # Pick Action
#         legalActions = self.getLegalActions(state)
#         action = None
#         "*** YOUR CODE HERE ***"
#         legalActions = self.getLegalActions(state)
#         if not legalActions:
#             return None

#         # Explore: With probability epsilon, take a random action
#         if util.flipCoin(self.epsilon):
#             return random.choice(legalActions)

#         # Exploit: Take the best action based on the current Q-values
#         print(self.epsilon)
#         return self.computeActionFromQValues(state)


#     def update(self, state, action, nextState, reward: float):
#         """
#           The parent class calls this to observe a
#           state = action => nextState and reward transition.
#           You should do your Q-Value update here
#           NOTE: You should never call this function,
#           it will be called on your behalf
#         """
#         "*** YOUR CODE HERE ***"
#         # Get the current Q-value
#         currentQValue = self.getQValue(state, action)
#         # Compute the future value based on nextState
#         nextValue = self.computeValueFromQValues(nextState)
#         # Update Q-value using the formula
#         updatedQValue = ((1 - self.alpha) * currentQValue) + (self.alpha * (reward + self.discount * nextValue))
#         # Store the updated Q-value
#         self.qValues[(state, action)] = updatedQValue

#     def getPolicy(self, state):
#         return self.computeActionFromQValues(state)

#     def getValue(self, state):
#         return self.computeValueFromQValues(state)


import csv

import csv

class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # Initialize Q-values and performance tracking
        self.qValues = util.Counter()  # A Counter is a dict with default 0
        self.episode_count = 0  # To track the number of episodes
        self.cumulative_reward = 0  # Cumulative reward for the current episode
        self.filename = "training_performance_decaying.csv"  # File to save performance data

        # Initialize the CSV file and write the header
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["episode", "cumulative_reward", "average_q_value", "epsilon"])
            writer.writeheader()

    def getQValue(self, state, action):
        return self.qValues[(state, action)]  # Default is 0 for unseen state-action pairs

    def computeValueFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        return max([self.getQValue(state, action) for action in legalActions])

    def computeActionFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        best_action = None
        best_value = float('-inf')

        for action in legalActions:
            q_value = self.getQValue(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action

        return best_action

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        # Update epsilon for exploration
        self.epsilon = max(0, self.epsilon * 0.999)  # Exponential decay with a floor of 0.9
        #print(self.epsilon)

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)

        return self.computeActionFromQValues(state)
        #bool= true
    def update(self, state, action, nextState, reward: float, done: bool):
        """
        Update Q-values and log performance data at the end of an episode.
        """
        # Update cumulative reward
        self.cumulative_reward += reward

        # Update Q-values
        currentQValue = self.getQValue(state, action)
        nextValue = self.computeValueFromQValues(nextState)
        updatedQValue = ((1 - self.alpha) * currentQValue) + (self.alpha * (reward + self.discount * nextValue))
        self.qValues[(state, action)] = updatedQValue

        # If the episode ends, save performance data
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

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
