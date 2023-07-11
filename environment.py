import gym
import numpy as np
import serial
import random

class MicroSwimmerEnvironment(gym.Env):
    def __init__(self):
        # Define environment parameters
        self.pool_width = 10
        self.pool_height = 10
        self.target_location = np.array([8, 8])
        self.current_location = np.array([2, 2])
        self.max_episode_steps = 15
        self.velocity = np.array([0.0, 0.0])
        self.steps = 0



        # Define state space and action space
        self.observation_space = gym.spaces.Dict({
            'current_location': gym.spaces.Box(low=0, high=self.pool_width, shape=(2,), dtype=np.float32),
            'target_location': gym.spaces.Box(low=0, high=self.pool_width, shape=(2,), dtype=np.float32),
            'velocity': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        })
        
        self.action_space = gym.spaces.Discrete(8)

    def step(self, action):
        # Convert action to movement
        direction = self._map_action_to_direction(action)
        movement = self._get_movement_from_direction(direction)

        # Calculate the distance to the target
        previous_distance = np.linalg.norm(self.current_location - self.target_location)

        # Calculate the velocity based on the chosen action
        velocity = movement / np.linalg.norm(movement)
        # Update velocity of the microswimmer
        self.velocity = velocity

        # Update microswimmer's current location
        new_location = self.current_location + movement * self.velocity

        # Calculate the distance to the target using the updated location
        distance_to_target = np.linalg.norm(self.target_location - new_location)
        current_distance = np.linalg.norm(new_location - self.target_location)

        # Calculate the reward
        reward = self._calculate_reward(previous_distance, current_distance)

        # Update the current location of the microswimmer
        self.current_location = new_location

        self.steps += 1

        # Determine if the episode is terminated
        done = (distance_to_target <= 1.5) or (self.steps >= self.max_episode_steps)

        # Construct the new state
        new_state = {
            'current_location': self.current_location,
            'target_location': self.target_location,
            'velocity': self.velocity
        }


        return new_state, reward, done, {}

    def reset(self):
        # Reset the location of the microswimmer
        self.current_location = np.array([2, 2])
        self.steps = 0
        self.velocity = np.array([0.0, 0.0])  # Reset the velocity

        # Return the initial state
        return {
            'current_location': self.current_location,
            'target_location': self.target_location,
            'velocity': self.velocity
        }

    def _map_action_to_direction(self, action):
        # Map action to direction
        directions = ['u', 'o', 'r', 'm', 'd', 'n', 'l', 'p']
        return directions[action]

    def _get_movement_from_direction(self, direction):
        # Calculate the movement based on the direction
        if direction == 'u':
            return np.array([0, 1])
        elif direction == 'o':
            return np.array([1, 1])
        elif direction == 'r':
            return np.array([1, 0])
        elif direction == 'm':
            return np.array([1, -1])
        elif direction == 'd':
            return np.array([0, -1])
        elif direction == 'n':
            return np.array([-1, -1])
        elif direction == 'l':
            return np.array([-1, 0])
        elif direction == 'p':
            return np.array([-1, 1])
        else:
            return np.array([0, 0])

    
    def _calculate_reward(self, previous_distance, current_distance):
        reward = 0.0

        if current_distance < previous_distance:
            reward += 1
        else:
            reward += -1

        if current_distance < 0.3:
            reward += 10

        return reward


    def render(self):
        pool_size = 10  # Size of the pool in the rendering

        # Create an empty pool image
        pool_img = np.zeros((self.pool_height, self.pool_width))

        # Draw the target location
        target_pos = (int(self.target_location[0]), int(self.target_location[1]))
        pool_img[target_pos[1], target_pos[0]] = 1.0

        # Convert the continuous position to integer indices
        microswimmer_pos = (int(self.current_location[0]), int(self.current_location[1]))
        microswimmer_pos = np.clip(microswimmer_pos, 0, pool_size - 1)  # Clip the indices to valid range

        # Draw the current location of the microswimmer
        pool_img[microswimmer_pos[1], microswimmer_pos[0]] = 0.5

        # Print the pool image
        for row in pool_img:
            for val in row:
                if val == 1.0:
                    print("T", end=" ")  # Target location
                elif val == 0.5:
                    print("M", end=" ")  # Microswimmer location
                else:
                    print(".", end=" ")  # Empty space
            print()


    


if __name__ == "__main__":
    env = MicroSwimmerEnvironment()

    ser = serial.Serial('/dev/cu.usbserial-10', 9600, timeout=1)
    
    episodes = 10

    for episode in range(episodes+1):
        state = env.reset()
        done = False
        score = 0
        steps = 0

        while not done:
            action = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)
            steps += 1
            score += reward

            print(f"Episode: {episode}, Steps: {steps}, Action: {action}, Reward: {score}")

            env.render()

            if done:
                break


# /dev/cu.usbserail-10