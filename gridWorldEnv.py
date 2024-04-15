#This code was initially taken from the gym official documents and adapted to the current project.
# link: https://www.gymlibrary.dev/content/environment_creation/

import gym
from gym import spaces
import pygame
import numpy as np
import random
from gymnasium.envs.registration import register

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=9):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.wall_locations = [ np.array([5,i]) for i in (0,2,3,4,5,6,7,8,9)
        ]
        # print("walls",self.wall_locations)
        
        self.observation_space = np.array(
                [(0,0,0,0),(self.size-1,self.size-1,self.size-1,self.size-1)]
                # "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                # "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            ).T

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self._action_to_direction_sideways = {
            0: [np.array([1, 1]), np.array([1, -1])],
            1: [np.array([1, 1]), np.array([-1, 1])],
            2: [np.array([-1, 1]), np.array([-1, -1])],
            3: [np.array([1, -1]), np.array([-1, -1])],
        }


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def switch_doors(self, top_door, bottom_door):
        self.wall_locations = [ np.array([5,i]) for i in (0,2,3,4,5,6,7,8)]
        if top_door=="close":
            self.wall_locations += [ np.array([5,1])]
        if bottom_door=="close":
            self.wall_locations  += [ np.array([5,9])]

    def _get_obs(self):
        return np.array([self._agent_location[0], self._agent_location[1], self._target_location[0],self._target_location[1]],  dtype=np.float32)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([0,0])
        # self.np_random.integers(0, self.size, size=2, dtype=int)
        # print(self._agent_location, type(self._agent_location))
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = np.array([9,0])
        # self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        direction_sideways = self._action_to_direction_sideways[action]
        # We use `np.clip` to make sure we don't leave the grid
        
        if random.random() < 0:
            new_location = np.clip(
            self._agent_location + random.choice(direction_sideways), 0, self.size - 1
            )
        else:
            new_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
            )
        
        # new_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )
        walls = list(map(lambda x: tuple(x), self.wall_locations))
        # print("tuple new location",tuple(new_location))
        # print("wall locaitons",walls)

        if tuple(new_location) not in walls:
            self._agent_location = new_location
        else:
            pass
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for wall_loc in self.wall_locations:
            pygame.draw.rect(
            canvas,
            (0, 0, 0),  # Color of the wall
            pygame.Rect(
                pix_square_size * wall_loc,
                (pix_square_size, pix_square_size),),
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# register(
#      id="gym_examples/myGridWorld-v0",
#      entry_point="gym_examples.envs:GridWorldEnv",
#      max_episode_steps=300,
# )