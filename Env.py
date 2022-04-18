# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0,0)] + [(i,j) for i in range(m) for j in range(m) if i != j]
        self.state_space = [(i,j,k) for i in range(m) for j in range(t) for k in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()

    def state_get_loc(self, state):
        return state[0]

    def state_get_time(self, state):
        return state[1]

    def state_get_day(self, state):
        return state[2]

    def action_get_pickup(self, action):
        return action[0]

    def action_get_drop(self, action):
        return action[1]

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""

        # print('STATE (in arch):', state)
        state_encod = np.zeros(m + t + d)
        state_encod[self.state_get_loc(state)] = 1
        state_encod[m + self.state_get_time(state)] = 1
        state_encod[m + t + self.state_get_day(state)] = 1

        return state_encod


    # Use this function if you are using architecture-2
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""


    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""
        location = self.state_get_loc(state)

        if location == 0:
            requests = np.random.poisson(2)

        elif location == 1:
            requests = np.random.poisson(12)

        elif location == 2:
            requests = np.random.poisson(4)

        elif location == 3:
            requests = np.random.poisson(7)

        elif location == 4:
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        actions.append((0,0))
        possible_actions_index.append(0) # index for (0,0) which is not considered as customer request

        return possible_actions_index, actions

    # def reward_func(self, state, action, Time_matrix):
    def reward_func(self, wait_time, transit_time, passenger_time):
        """
        The reward function for the driver.
        The reward is constant R times the time passenger was in the car
        minus the cost of fuel consumed in the car times C
        """
        reward = R*passenger_time - C*(passenger_time+wait_time+transit_time)

        return reward

    def update_time_day(self, time, day, ride_duration):
        """
        Takes in the current state and time taken for driver's journey to return
        the state post that journey.
        """
        ride_duration = int(ride_duration)

        if (time + ride_duration) < 24:
            # day is unchanged
            time = time + ride_duration

        else:
            # duration taken spreads over to subsequent days
            # convert the time to 0-23 range
            time = (time + ride_duration) % 24

            # Get the number of days
            num_days = (time + ride_duration) // 24

            # Convert the day to 0-6 range
            day = (day + num_days) % 7

        return time, day


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state

         3 Scenarios:
           a) Refuse all requests
           b) Driver is already at pick up point
           c) Driver is not at the pickup point.
        """
        next_state = []

        # Initialize various times
        total_time   = 0
        transit_time = 0    # to go from current location to pickup location
        wait_time    = 0    # in case driver chooses to refuse all requests
        ride_time    = 0    # from Pick-up to drop

        # Derive the current location, time, day and request locations
        curr_loc = self.state_get_loc(state)
        pickup_loc = self.action_get_pickup(action)
        drop_loc = self.action_get_drop(action)
        curr_time = self.state_get_time(state)
        curr_day = self.state_get_day(state)

        if pickup_loc == 0 and drop_loc == 0:
            # Refuse all requests, so wait time is 1 unit, next location is current location
            wait_time = 1
            next_loc = curr_loc

        elif curr_loc == pickup_loc:
            # means driver is already at pickup point, wait and transit are both 0 then.
            ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]

            # next location is the drop location
            next_loc = drop_loc

        else:
            # Driver is not at the pickup point, he needs to travel to pickup point first
            # time take to reach pickup point
            transit_time      = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            new_time, new_day = self.update_time_day(curr_time, curr_day, transit_time)

            # The driver is now at the pickup point
            # Time taken to drop the passenger
            ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]
            next_loc  = drop_loc

        # Calculate total time as sum of all durations
        total_time = wait_time + transit_time + ride_time
        next_time, next_day = self.update_time_day(curr_time, curr_day, total_time)

        # Construct next_state using the next_loc and the new time states.
        next_state = [next_loc, next_time, next_day]

        return next_state, wait_time, transit_time, ride_time

    def step(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state and reward"""

        # Get the next state using the current state and action
        next_state, wait_time, transit_time, ride_time = self.next_state_func(state, action, Time_matrix)

        # Calculate the reward as a function of wait_time, transit_time and ride_time
        rewards = self.reward_func(wait_time, transit_time, ride_time)

        total_time = wait_time + transit_time + ride_time

        return rewards, next_state, total_time


    def reset(self):
        return self.action_space, self.state_space, self.state_init


    def test_run(self):
        """
        This fuction can be used to test the environment
        """
        # Loading the time matrix provided
        import operator
        Time_matrix = np.load("TM.npy")
        print("CURRENT STATE: {}".format(self.state_init))

        # Check request at the init state
        requests = self.requests(self.state_init)
        print("REQUESTS: {}".format(requests))

        # # compute rewards
        # rewards = []
        # for req in requests[1]:
        #     r =  self.reward_func(self.state_init, req, Time_matrix)
        #     rewards.append(r)
        # print("REWARDS: {}".format(rewards))

        new_states = []
        rewards = []
        for req in requests[1]:
            s = self.next_state_func(self.state_init, req, Time_matrix)
            new_states.append(s[0])

            r = self.reward_func(s[1],s[2],s[3])
            rewards.append(r)

        print("NEW POSSIBLE STATES: {}".format(new_states))
        print("REWARDS: {}".format(rewards))

        # if we decide the new state based on max reward
        index, max_reward = max(enumerate(rewards), key=operator.itemgetter(1))
        self.state_init = new_states[index]
        print("MAXIMUM REWARD: {}".format(max_reward))
        print ("ACTION : {}".format(requests[1][index]))
        print("NEW STATE: {}".format(self.state_init))
        print("NN INPUT LAYER (ARC-1): {}".format(self.state_encod_arch1(self.state_init)))

if __name__ == "__main__":
    env = CabDriver()
    env.test_run()
