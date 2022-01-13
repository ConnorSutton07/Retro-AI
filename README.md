# Retro-AI

This project provides an framework for implementing and testing Deep Reinforcement Learning algorithms in Super Nintendo games. It is ideal for beginners trying to build up an intuition as to the mechanics of Deep RL or for experts that want to test new algorithms or experiment with transfer learning. 

![small_mario](https://user-images.githubusercontent.com/55513603/120692158-020fc780-c46d-11eb-8205-cd834f1ece4e.png)
![small_dk](https://user-images.githubusercontent.com/55513603/120692160-020fc780-c46d-11eb-9d0f-64845703097b.png)



## To Do:
- Add game music/audio in spectate mode 
- Ability to choose game level
- More RL algorithms (PPO, Curiosity, Neuroevolution, etc.)
- Race against AI/Multiplayer
- Add interactive mode
- Add documentation

## Setup

### Step 1
Clone this repo and setup a Python 3.7 environment--other versions of Python *will not work*.


### Step 2
run ``` pip install -r requirements.txt ```

### Step 3 
Gym-Retro comes with a 'scenario.json' and 'data.json' file for each of the hundreds of games for which in an interface is setup. The data files create variables to track game elements such as the player's x/y position, score, number of lives, etc. The scenario file allows you to specify how the agent will be rewarded or when an episode should terminate based on the variables from the data file. These will often be lacking in content, however. For example, the default scenario.json file for Donkey Kong does not contain a variable to track the player's x position, which is essential for a decent reward function. Using Retro's Game Integration Tool, these can be added manually.

Open the 'Data' folder, and copy all the contained folders. Navigate to 'Lib\site-packages\retro\data\stable', and paste these folders to overwrite the default files with the updated versions.
