
def SMW_Reward(info: dict, prev_info: dict) -> int:
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

rewards = {
    'SuperMarioWorld-Snes' : SMW_Reward
}