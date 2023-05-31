import os.path
import random

import numpy as np

# 2 states:
#     atSpawn
#     onBoard

# 9 actions:
#     MoveOut
#     MoveDiceValue
#     Goal
#     Star
#     Globe
#     Protect
#     Kill
#     Die
#     GoGoalZone
#     GoGoal


class Rewards():
    UNABLE = -3
    BAD = -1
    NEUTRAL = 0.0
    OK = 0.25
    GOOD = 0.5
    VERY_GOOD = 1
    THE_BEST = 2

    # UNABLE = -100
    # VERY_BAD = -5.0
    # BAD = -2
    # NEUTRAL = 0.0
    # OK = 1
    # GOOD = 2
    # VERY_GOOD = 5.0
    # THE_BEST = 100

    def __init__(self):

        self.rewards_table = {
            'atSpawn_MoveOut' : self.VERY_GOOD,
            'atSpawn_MoveDiceValue' : self.UNABLE,
            'atSpawn_Star' : self.UNABLE,
            'atSpawn_Globe' : self.UNABLE,
            'atSpawn_Protect' : self.UNABLE,
            'atSpawn_Kill' : self.UNABLE,
            'atSpawn_Die' : self.UNABLE,
            'atSpawn_GoGoalZone' : self.UNABLE,
            'atSpawn_GoGoal' : self.UNABLE,
            
            'onBoard_MoveOut' : self.UNABLE,
            'onBoard_MoveDice' : self.NEUTRAL,
            'onBoard_Star' : self.OK,
            'onBoard_Globe' : self.OK,
            'onBoard_Protect' : self.GOOD,
            'onBoard_Kill' : self.VERY_GOOD,
            'onBoard_Die' : self.BAD,
            'onBoard_GoGoalZone' : self.VERY_GOOD,
            'onBoard_GoGoal' : self.THE_BEST,
        }

    def __str__(self):
        return f"{self.rewards_table}"
    
    def getReward(self, key):
        return self.rewards_table[key]
    
    def setRewards(self, newrewards):
        self.rewards_table = newrewards