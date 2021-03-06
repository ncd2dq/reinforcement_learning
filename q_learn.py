import numpy as np
import time
import os
from math import sqrt

# Array indexing [row:col]

def create_world():
    '''
    0 is a viable path
    1 is an inviable path
    '''
    world = np.array([
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
                        [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
                        [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
                        [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                        [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ])

    return world

class Agent(object):
    def __init__(self, initial_state):
        self.state = initial_state
        self.visual = 7


def instantiate_q_dict(world):
    '''
    ::param world:: numpy square array

    returns a q_dict --> {(row, col): [left, right, up, down]}
    '''
    q_dict = {}
    for i, row in enumerate(world):
        for j, col in enumerate(row):
            location = (i, j)
            actions = [0, 0, 0, 0]
            q_dict[location] = actions

    return q_dict


def choose_action(q_dict, state, greed_policy=0.85):
    '''
    ::param state:: tuple of agent's current position (row, col)
    ::param greed_policy:: float of the probability that the agent will take the current "best" path

    return 0 1 2 3 for left right up down
    '''
    rand_val = np.random.random()
    if rand_val < greed_policy:
        # Take the "best" path
        if all(elm == 0 for elm in q_dict[state]):
            # If they're all 0, just randomly pick
            action = np.random.choice( [i for i in range( len( q_dict[state] ) )] )
        else :
            # Take best path
            action = q_dict[state].index(max(q_dict[state]))
    else:
        # Choose a random action 
        action = np.random.choice( [i for i in range( len( q_dict[state] ) )] )

    if action == 0:

        return (0, -1)

    elif action == 1:

        return (0, 1)

    elif action == 2:

        return (-1, 0)

    elif action == 3:

        return (1, 0)


def _get_new_state_world_val(world, state, action):
    new_state = [state[0] + action[0], 
                state[1] + action[1]]

    return world[new_state[0]][new_state[1]]

def validate_action(world, state, action, invalid=1):
    '''
    ::param world:: numpy square array
    ::param state:: tuple of current position (row, col)
    ::param action:: tuple of row/col change 
    (1, 0) -> down
    (-1, 0) -> up
    (0, 1) -> right
    (0, -1) -> left
    ::param valid:: the value in the array that represents an invalid move

    return boolean if move is possible
    '''

    if _get_new_state_world_val(world, state, action) == invalid:

        return False

    return True


def _dist_to_target(state, target):
    '''
    Return distance to target
    '''
    under = (target[0] - state[0])**2 + (target[1] - state[1])**2
    return sqrt(under)

def decide_reward(world, state, action, target, q_dict, win=2, discount=0.9):
    
    # If move is invalid, heavily penalize
    if validate_action(world, state, action) is False:

        return (-1)


    # At the new state, you get part of the Q value of the best move in the new state
    discounted_future_reward = discount * max(q_dict[(state[0] + action[0], state[1] + action[1])])
    # If position is the target, heavily reward
    # TODO store all moves, if the thing ends up winning, reward all moves better than if it doesn't
    if _get_new_state_world_val(world, state, action) == win:

        return 100

    # NO HELP
    # # If it gets closer to the target reward
    # elif _dist_to_target(state, target) < _dist_to_target((state[0] + action[0], state[1] + action[1]), target):

    #     return 1 + discounted_future_reward

    # No reward if no progress
    else:
        # Punish slightly if not correct space to minimize number of moves
        return 0 + discounted_future_reward - 0.01


def update_q_dict(state, action, reward, q_dict, learning_rate=0.1):
    # to account for future reward,
    # Each states Q should be it's accumulated reward
    # plus a some amount of the Q value of the best action
    # in the next state
    if action == (0, -1):
        index = 0

    elif action == (0, 1):
        index = 1

    elif action == (-1, 0):
        index = 2

    elif action == (1, 0):
        index = 3

    q_dict[state][index] += reward * learning_rate

    return q_dict


def training():
    count = 0
    agent = Agent((1, 1))
    world = create_world()
    q_dict = instantiate_q_dict(world)
    target = [11, 10]


    # TODO give a negative reward every move to minimize # of moves
    # TODO Q value decay
    # For displaying the map, put agent on map and store value
    # Move agent, replace old value, store new value
    stored_value = [[0, 0], 0]
    complete = 0
    moves_array = []
    # Slowly increase greedy factor
    # Maximum exploration at first, then maximum exploitation by end
    greedy = 0.7
    remainder = 1 - greedy
    increment = remainder / 95
    while complete < 200:
        if complete % 10 == 0 and complete != 0: #only display game every 5 games
            stored_value = [[agent.state[0], agent.state[1]], world[agent.state[0]][agent.state[1]]]
            world[agent.state[0]][agent.state[1]] = agent.visual
            # Clear the terminal before new frame
            os.system('cls' if os.name == 'nt' else 'clear')
            print(world)
            world[stored_value[0][0]][stored_value[0][1]] = stored_value[1]
            print('Stage: ', complete)
            print('Greedy Policy: ', greedy)
        # Action returned as tuple with position deltas (-1, 0)

        action = choose_action(q_dict, agent.state, greed_policy=greedy)
        while not validate_action(world, agent.state, action):
            action = choose_action(q_dict, agent.state)

        # Reward returned as a number
        reward = decide_reward(world, agent.state, action, target, q_dict)

        q_dict = update_q_dict(agent.state, action, reward, q_dict)

        if _get_new_state_world_val(world, agent.state, action) == 2: #terminal state
            agent.state = (1, 1)
            complete += 1
            if greedy < 1.00:
                greedy += increment
            moves_array.append(count)
            count = 0
            time.sleep(2)
        else:
            agent.state = (agent.state[0] + action[0], agent.state[1] + action[1])

        # TODO, when agent wins, put him back at the start
        count += 1

    print(moves_array)
    msg = ''
    while msg != 'q':
        msg = input('Did you finish?')

training()
