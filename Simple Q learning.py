"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 5   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move


def build_q_table(n_states, actions):
    '''
    Creates a table of zeros with # rows based on states and # columns based on actions.
    the value of the number in for each state(row) and action(column) is the likely hood that
    it will be picked at that state.
    '''
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values 1 row for each state, and 1 column for each possible action
        columns=actions,    # actions's name
    )

    #  [State, Left Right]
    #  [0       0    0   ]
    #  [1       0    0   ]
    # The numbers are the probability that it will take that action for that state, numbers are updated by algorithm
    
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]  # the state is the position on the 1D axis
    # iloc, at state 3, return all columns (all possible actions at that state)
    # ^ state action is, for example   3 0.1 0.9 (at position with index 3, 10% chance go left, 90% chance go right)
    
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS) # if it's the very first iteration (all 0 state-action) or being non-greedy,
        # all have equal chance of being chosen as the action (basically ignore past information)
        
    else:   # act greedy (take the highest probability)
        action_name = state_actions.argmax() # do the action that has the highest Q value
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    '''recieves the current state, and the action we decided to take'''
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate: N_STATES - 1 is the end, so if we are just before the end, and move right,
                                # we reached end
            S_ = 'terminal'
            R = 1  # Only give a reward if we reached the end on that step
        else:
            S_ = S + 1
            R = 0
            
    else:   # move left
        R = 0 # No reward is given because we want to move right
        if S == 0:
            S_ = S  # If State == 0 we are at the left most wall and do not want to move, so the next state, S_,
                    # is still 0
        else:
            S_ = S - 1  #If we are not at the left most wall (S == 0), then move left
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    '''This controls what is printed to the screen'''
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS) # create your Q table based on states and action parameters
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter) # draw env to screen
        while not is_terminated:

            A = choose_action(S, q_table) # choose action based on what actions are possible at our state
            S_, R = get_env_feedback(S, A)  # Using the A(action) we chose at this S(state), take action & get next state and reward
            
            q_predict = q_table.ix[S, A] # ix allows you to either give an index or a column label when finding a value,
            #                              returns Q value
            
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
                # either target is really small, or the added Reward makes it substantial
                
            else: #the next step is terminal
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.ix[S, A] += ALPHA * (q_target - q_predict)  # target is either 0 or 1,
            #                                     add to the Q value by a Learning Rate * difference of current Q value and target
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
