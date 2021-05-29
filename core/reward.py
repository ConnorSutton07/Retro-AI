
def SMW_Reward(info: dict, prev_info: dict) -> int:
    """
    Simple reward function designed for Super Mario World (SNES)
    Returns a reward value based on distance, score, death,
    and completing the level

    """
    if not prev_info is None:
        score = info['score'] - prev_info['score']
        xPos = info['x'] - prev_info['x']
        dead = info['dead']
        endOfLevel = info['endOfLevel']
        done = False
        reward = 0
        # Add to fitness score if mario gains points on his score.
         # If mario dies, dead becomes 0, so when it is 0, penalize him and move on.
        if dead == 0:
            reward -= 100
            done = True
            return reward, done
        if endOfLevel == 1:
            reward += 500
            done = True
            return reward, done

        #print("Prev:", prev_info['x'], "| Current:", info['x'], "| Reward:", xPos)
        reward = xPos
        reward += score // 100
        return reward, done
    else:
        return 0, False


def DKC_Reward(info: dict, prev_info: dict) -> int:
    if not prev_info is None:
        score = info['score'] - prev_info['score']
        dead = info['lives'] < prev_info['lives']
        reward = score + (-100 * dead)
        return reward, dead
    else:
        return 0, False

def default_reward_function(info: dict, prev_info: dict, reward: int) -> int:
    return reward, false


Rewards = {
    'SuperMarioWorld-Snes' : SMW_Reward,
    'DonkeyKongCountry-Snes' : DKC_Reward
}