import random


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

class Qtable():

    def __init__(self):

        self.qtable = {
            'atSpawn_MoveOut' : 0,
            'atSpawn_MoveDiceValue' : 0,
            'atSpawn_Star' : 0,
            'atSpawn_Globe' : 0,
            'atSpawn_Protect' : 0,
            'atSpawn_Kill' : 0,
            'atSpawn_Die' : 0,
            'atSpawn_GoGoalZone' : 0,
            'atSpawn_GoGoal' : 0,
            
            'onBoard_MoveOut' : 0,
            'onBoard_MoveDice' : 0,
            'onBoard_Star' : 0,
            'onBoard_Globe' : 0,
            'onBoard_Protect' : 0,
            'onBoard_Kill' : 0,
            'onBoard_Die' : 0,
            'onBoard_GoGoalZone' : 0,
            'onBoard_GoGoal' : 0,
        }
        
        # {
            # 'atSpawn_MoveOut' : 2,
            # 'atSpawn_MoveDiceValue' : -1,
            # 'atSpawn_Star' : -1,
            # 'atSpawn_Globe' : -1,
            # 'atSpawn_Protect' : -1,
            # 'atSpawn_Kill' : -1,
            # 'atSpawn_Die' : -1,
            # 'atSpawn_GoGoalZone' : -1,
            # 'atSpawn_GoGoal' : -1,
            # 
            # 'onBoard_MoveOut' : -1,
            # 'onBoard_MoveDice' : 0.3,
            # 'onBoard_Star' : 0.5,
            # 'onBoard_Globe' : 0.5,
            # 'onBoard_Protect' : 0.7,
            # 'onBoard_Kill' : 1,
            # 'onBoard_Die' : -1,
            # 'onBoard_GoGoalZone' : 0.8,
            # 'onBoard_GoGoal' : 5,
        # }
# 
        self.standard = {
            'atSpawn_MoveOut' : 0,
            'atSpawn_MoveDiceValue' : 0,
            'atSpawn_Star' : 0,
            'atSpawn_Globe' : 0,
            'atSpawn_Protect' : 0,
            'atSpawn_Kill' : 0,
            'atSpawn_Die' : 0,
            'atSpawn_GoGoalZone' : 0,
            'atSpawn_GoGoal' : 0,
            
            'onBoard_MoveOut' : 0,
            'onBoard_MoveDice' : 0,
            'onBoard_Star' : 0,
            'onBoard_Globe' : 0,
            'onBoard_Protect' : 0,
            'onBoard_Kill' : 0,
            'onBoard_Die' : 0,
            'onBoard_GoGoalZone' : 0,
            'onBoard_GoGoal' : 0,
        }

    def __str__(self):
        return f"{self.qtable}"
    
    def getTable(self):
        return self.qtable
    
    def getValue(self, key):
        return self.qtable[key]
    
    def reset(self):
        self.qtable = self.standard
    
    def updateQtable(self, key, reward, next_best, gamma, learning_rate):
        self.qtable[key] = learning_rate * (reward + gamma*self.qtable[next_best] - self.qtable[key])

    def setQtable(self, newqtable):
        self.qtable = newqtable
    