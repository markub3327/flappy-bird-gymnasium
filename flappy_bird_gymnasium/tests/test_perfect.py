# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
# Copyright (c) 2023 Martin Kubovcik
# Copyright (c) 2024 Akshay Gulabrao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""Tests the simple-observations version of the Flappy Bird environment with a
human player.
"""

import gymnasium
import flappy_bird_gymnasium


def agent(obs):
    pipe = 0
    if obs[0] < 5:
        pipe = 1
    x = obs[pipe * 3]
    bot = obs[pipe * 3 + 2]
    top = obs[pipe * 3 + 1]
    y_next = obs[-3] + obs[-2] + 24 + 1
    if 74 < x < 88 and obs[-3] - 45 >= top:
        return 1
    elif y_next >= bot:
        return 1
    return 0


def play(use_lidar=True, render_mode="human"):
    env = gymnasium.make(
        "FlappyBird-v0",
        audio_on=True,
        render_mode=render_mode,
        use_lidar=use_lidar,
        normalize_obs=False,
        score_limit=1000,
    )

    steps = 0
    video_buffer = []

    obs, _ = env.reset()
    while True:
        # Getting action:
        action = agent(obs)
        # Processing:
        obs, reward, done, term, info = env.step(action)
        video_buffer.append(obs)

        steps += 1

        if done or term:
            break

    env.close()
    return info["score"]


if __name__ == "__main__":
    play(use_lidar=False, render_mode="human")
