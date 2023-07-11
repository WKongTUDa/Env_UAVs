import random, math
import numpy as np
import matplotlib.pyplot as plt

from pettingzoo.utils import ParallelEnv
from gym import spaces
from PIL import Image

import os

from config import grid_size, n_agents, n_obstacles, observation_shape

class ParallelGridGame(ParallelEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.observation_shape = observation_shape

        self.possible_agents = ["agent_" + str(i) for i in range(n_agents)]
        self.agent_status = ["active" for _ in range(self.n_agents)]

        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents if agent != "null"}
        self.observation_spaces = {agent: spaces.Box(low=0, high=2, shape=observation_shape) for agent in self.possible_agents if agent != "null"}


        self.agent_positions = [(i, 0) for i in range(n_agents)]
        self.agent_selection = 0

        self.obstacle_positions = self._generate_obstacles()

        self.grid = np.zeros(grid_size)
        self.grid_observed = np.zeros(grid_size)
        self.total_observed = 0
        self.total_grid_num = np.prod(grid_size)

        self.wind_direction = (1, 1)
        # self.agent_battery = [50 for _ in range(self.n_agents)]
        self.agent_battery = [random.randint(35, 50) for _ in range(self.n_agents)]
        
        self.render_counter = 0
        self.step_count = 0

    def _agent_selector(self):
        return self.agents[self.agent_selection]

    def _observe(self):
        agent_position = self.agent_positions[self.agent_selection]
        agent_observation = self._get_agent_observation(self.agent_selection)
        return {'agent_observation': agent_observation, 'agent_position': agent_position}

    def _get_agent_observation(self, agent_idx):
        agent_pos = self.agent_positions[agent_idx]
        agent_row, agent_col = agent_pos
        grid = np.zeros(self.observation_shape)

        for i in range(agent_row - 1, agent_row + 2):
            for j in range(agent_col - 1, agent_col + 2):
                if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:
                    if (i, j) in self.obstacle_positions:  
                        grid[i - agent_row + 1][j - agent_col + 1] = 1
                    elif (i, j) in self.agent_positions:
                        grid[i - agent_row + 1][j - agent_col + 1] = 2
                    else:
                        grid[i - agent_row + 1][j - agent_col + 1] = 0

        return grid.flatten()

    def _observe_grid(self, agent_idx):
        agent_position = self.agent_positions[agent_idx]
        for i in range(agent_position[0] - 1, agent_position[0] + 2):
            for j in range(agent_position[1] - 1, agent_position[1] + 2):
                if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:
                    if self.grid_observed[i, j] == 0:
                        self.grid_observed[i, j] = 1
                        self.total_observed += 1

    def _generate_obstacles(self): 
        obstacle_positions = []
        while len(obstacle_positions) < self.n_obstacles:
            x = np.random.randint(0, self.grid_size[0])
            y = np.random.randint(0, self.grid_size[1])
            position = (x, y)
            if position not in obstacle_positions and position not in [(i, 0) for i in range(n_agents)]:
                obstacle_positions.append(position)
        return obstacle_positions
    
    def _check_collision(self, agent_idx, new_position):
        
        for i, position in enumerate(self.agent_positions):
            if i != agent_idx and position == new_position:
                return True

        for obstacle_position in self.obstacle_positions:
            if obstacle_position == new_position:
                return True

        return False
    
    def _calculate_new_position(self, agent_position, action):

        if action == 0:  # up
            new_position = (agent_position[0] - 1, agent_position[1])
        elif action == 1:  # down
            new_position = (agent_position[0] + 1, agent_position[1])
        elif action == 2:  # left
            new_position = (agent_position[0], agent_position[1] - 1)
        elif action == 3:  # right
            new_position = (agent_position[0], agent_position[1] + 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check bounds 
        if new_position[0] < 0:
            new_position = (0, new_position[1])
        elif new_position[0] >= self.grid_size[0]:
            new_position = (self.grid_size[0] - 1, new_position[1])
        if new_position[1] < 0:
            new_position = (new_position[0], 0)
        elif new_position[1] >= self.grid_size[1]:
            new_position = (new_position[0], self.grid_size[1] - 1)

        return new_position

    '''
    def _perform(self, action):
        agent_position = self.agent_positions[self.agent_selection]
        new_position = self._calculate_new_position(agent_position, action)
        
        if self._check_collision(new_position):
            # If collision, reset agent position and stop exploration
            self.agent_positions[self.agent_selection] = agent_position
            return
        
        self.agent_positions[self.agent_selection] = new_position
    '''

    def reset(self, seed=None, options=None):

        self.agents = self.possible_agents
        self.agent_status = ["active" for _ in range(self.n_agents)]

        self.agent_positions = [(i, 0) for i in range(self.n_agents)]
        self.agent_selection = 0

        self.grid_observed = np.zeros(self.grid_size)
        self.total_observed = 0

        for agent_idx in range(self.n_agents):
            self._observe_grid(agent_idx)

        self.agent_battery = [random.randint(35, 50) for _ in range(self.n_agents)]
        self.step_count = 0

        self.render_counter = 0

        observations = {agent: self._get_agent_observation(agent_idx) for agent_idx, agent in enumerate(self.agents)}

        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        reward = 0
        done = False

        for agent_idx, action in enumerate(actions):

            if self.agent_status[agent_idx] == "stopped":
                continue
            
            if self.agent_battery[agent_idx] < 5:
                self.agent_status[agent_idx] = "stopped"
                self.agents[agent_idx] = "null"
                continue

            agent_position = self.agent_positions[agent_idx]
            new_position = self._calculate_new_position(agent_position, action)

            if self._check_collision(agent_idx, new_position):
                self.agent_status[agent_idx] = "stopped"
                self.agents[agent_idx] = "null"
            else:
                # Calculate the angle between the direction of the agent's movement and the direction of the wind field
                dx = new_position[0] - agent_position[0]
                dy = new_position[1] - agent_position[1]

                if dx == 0 and dy == 0:
                    energy_cost = 0
                else:
                    motion_angle = math.degrees(math.atan2(dy, dx))
                    wind_angle = math.degrees(math.atan2(self.wind_direction[1], self.wind_direction[0]))
                    angle_diff = abs(motion_angle - wind_angle)

                    # Determine the power consumption according to the included angle
                    if angle_diff == 45:
                        energy_cost = 0.5
                    elif angle_diff == 135:
                        energy_cost = 1.5
                    else:
                        energy_cost = 1.0

                self.agent_positions[agent_idx] = new_position
                self.agent_battery[agent_idx] -= energy_cost
                self._observe_grid(agent_idx)

        # Check termination conditions
        terminations = [False for _ in range(self.n_agents)]
        if self.total_observed >= self.total_grid_num:
            reward = self._get_reward()
            terminations = [True for _ in range(self.n_agents)]
            self.agents = []
            done = True

        # Check truncation conditions
        truncations = [False for _ in range(self.n_agents)]
        if self.step_count >= 50:
            reward = self._get_reward()
            truncations = [True for _ in range(self.n_agents)]
            self.agents = []
            done = True

        done = done or all(agent == "null" for agent in self.agents)
        reward = self._get_reward()

        self.step_count += 1

        # Get observations
        # observations = [self._get_agent_observation(agent_idx) for agent_idx in range(self.n_agents)]
        observations = {agent: self._get_agent_observation(agent_idx) for agent_idx, agent in enumerate(self.agents)}

        infos = {agent: {} for agent in self.agents}

        return observations, reward, done, terminations, truncations, infos

    def render(self, save_dir=None):

        grid = np.zeros(self.grid_size, dtype=int)

        # Mark the observed grid as -1 (used to distinguish between observed and unobserved)
        grid[self.grid_observed == 1] = -1

        # Define colormap
        cmap = plt.cm.get_cmap('YlGn') 

        '''
        # Create images of agents and obstacles
        img_agents = Image.open(".agent_image.jpg")  
        img_obstacle = Image.open(".obstacle_image.jpg")  
        '''

        # Get the absolute path of the directory where the program is located
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the relative path of the image file
        agents_image_path = os.path.join(base_dir, "image", "agent_image.jpg")
        obstacles_image_path = os.path.join(base_dir, "image", "obstacle_image.jpg")

        # Open image file
        img_agents = Image.open(agents_image_path)
        img_obstacle = Image.open(obstacles_image_path)

        # Draw grid environment
        plt.figure(figsize=self.grid_size)
        plt.imshow(grid, cmap=cmap)

        # Mark the location of agents
        for agent_pos in self.agent_positions:
            plt.imshow(img_agents, extent=[agent_pos[1] - 0.5, agent_pos[1] + 0.5, agent_pos[0] - 0.5, agent_pos[0] + 0.5])

        # Mark the location of obstacles
        for obs_pos in self.obstacle_positions:
            plt.imshow(img_obstacle, extent=[obs_pos[1] - 0.5, obs_pos[1] + 0.5, obs_pos[0] - 0.5, obs_pos[0] + 0.5])

        plt.xticks([])
        plt.yticks([])
        plt.grid(color='gray', linewidth=0.5)

        # Visualize the observation space of each agent
        for agent_pos in self.agent_positions:
            row, col = agent_pos
            for i in range(row - 1, row + 2):
                for j in range(col - 1, col + 2):
                    if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:
                        plt.plot(j, i, '.', color=cmap(0.3), markersize=12)

        # Draw agent power
        for agent_pos, battery in zip(self.agent_positions, self.agent_battery):
            plt.text(agent_pos[1], agent_pos[0], str(battery), color='red', ha='center', va='center')

        # Draw wind arrow
        wind_direction = self.wind_direction
        for i in range(self.grid_size[0]+1):
            plt.arrow(-1.5, -1.5 + i, 0.5*wind_direction[1], 0.5*wind_direction[0], head_width=0.3, head_length=0.3, fc='gold', ec='gold')
            plt.arrow(-1.5 + i, -1.5, 0.5*wind_direction[1], 0.5*wind_direction[0], head_width=0.3, head_length=0.3, fc='gold', ec='gold')
        plt.arrow(-1.5, -1.5, 0.5*wind_direction[1], 0.5*wind_direction[0], head_width=0.3, head_length=0.3, fc='gold', ec='gold')

        plt.gca().invert_yaxis()

        # Save the plot to a file if save_dir is provided
        if save_dir:
            # Create the directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            # Generate a unique file name
            filename = f"image_{self.render_counter}.png"
            save_path = os.path.join(save_dir, filename)
            # Save the plot to the specified path
            plt.savefig(save_path)
            plt.close() 

        self.render_counter += 1

        # plt.show()

    def _is_done(self):
        return self.step_count >= 50 or self.total_observed >= self.total_grid_num or all(agent == "null" for agent in self.agents)

    def _get_reward(self):
        if self._is_done():
            exploration_ratio = self.total_observed / self.total_grid_num
            return exploration_ratio
        else:
            return 0
        
    def observation_space(self, agent):
        if agent != "null":
            return spaces.Box(low=0, high=2, shape=observation_shape)

    def action_space(self, agent):
        if agent != "null":
            return spaces.Discrete(4)
    