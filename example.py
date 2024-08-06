import numpy as np
import pdb
from env.snakes import SnakeEatBeans
from submissions.red import policy as red_policy
from submissions.blue import policy as blue_policy


env = SnakeEatBeans()
obs = env.reset(render=True)


action_dim = env.get_action_dim()
num_player = len(env.players)



while not env.is_terminal():
    
    action_red = red_policy(obs[:3])
    action_blue = blue_policy(obs[3:])

    all_actions = action_red + action_blue

    next_obs, reward, terminal, info = env.step(all_actions)

    
    
print(env.check_win())

