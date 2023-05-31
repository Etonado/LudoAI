import random
import numpy as np
import ludopy
from Ditlev.rewards import Rewards
from Ditlev.qtable import Qtable
import cv2
import copy
import matplotlib.pyplot as plt
import csv

def movingAverage(values, window_size):
    i = 0
    moving_averages = []
    
    while i < len(values) - window_size + 1:

        window = values[i : i + window_size]
        window_average = round(sum(window) / window_size, 2)
        moving_averages.append(window_average)
        i += 1
    
    return moving_averages



class Agent:
    star_positions = [5, 12, 18, 25, 31, 38, 44, 51]
    globe_positions_global = [9, 22, 35, 48]
    goal_zone_positions = [52, 53, 54, 55, 56]
    danger_positions = [14, 27, 40]
    all_globe_positions = globe_positions_global + danger_positions
    end_position = 57
    # def __init__(self, num_states=2, num_actions=9, learning_rate=0.1, reward_decay=0.2, epsilon_decay=0.01, e_greedy=1, id = 2):
    def __init__(self, num_states=2, num_actions=9, learning_rate=0.1, reward_decay=0.2, epsilon_decay=0.01, e_greedy=1, id = 2):
        # numpy table with dimensions num_statesxnum_actions
        self.number_states = num_states
        self.number_actions = num_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e_greedy = e_greedy
        self.epsilon = e_greedy
        self.id = id
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.001

        self.qtable = Qtable()
        self.rewards = Rewards()


    def __str__(self):
        return f"{self.qtable}"
    
    def setLR(self, newLR):
        self.lr = newLR
    
    def setRewardDecay(self, newRD):
        self.gamma = newRD

    def getQtable(self):
        return self.qtable.getTable()
    
    def setQtable(self, filename):
        with open(filename, mode='r') as infile:
            reader = csv.reader(infile)
            mydict = {rows[0]:rows[1] for rows in reader}
        self.qtable.setQtable(mydict)

    def setRewards(self, newre):
        self.rewards.setRewards(newre)
    
    def willLandOnGlobe(self, position):
        if position in self.globe_positions_global:
            return True
        else:
            return False
        
    def willLandOnStar(self, position):
        if position in self.star_positions:
            return True
        else:
            return False
        

    def pieceCanDie(self, piece, player_pieces, enemy_pieces, dice):
        
        next_position = player_pieces[piece] + dice
        amount_of_opponents_on_next_position = np.count_nonzero(enemy_pieces == next_position)

        # is there an enemy on a globe in the next position?
        if next_position in self.all_globe_positions and next_position in enemy_pieces:
            return True

        # if there are two or more opponents on next position
        if amount_of_opponents_on_next_position > 1:
            return True

        return False    


    def canMoveOut(self, player_pieces, move_pieces, dice):
        available_moves = []
        # check if a piece is in spawn
        if 0 in player_pieces and dice == 6:
            # index of piece in player_pieces array
            index = np.where(player_pieces==0)[0][0]

            # if it can be moved return the possible action with the index of the piece
            if index in move_pieces:
                available_moves.append(("atSpawn_MoveOut", index))
        #print("------  canMoveOut")
        return np.array(available_moves)
    

    def canMoveDice(self, player_pieces, move_pieces, enemy_pieces, dice):
        
        available_moves = []
        # finding all pieces that can be moved
        for i, position in enumerate(player_pieces):
            # checking if the piece is in move_pieces
            if i in move_pieces:
                if position+dice not in enemy_pieces and not self.willLandOnGlobe(position=position+dice) and not self.willLandOnStar(position=position+dice) and position != 0:
                    # i there is not an enemy on the position it will move
                    # and if it is not a star or globe is it added to available move
                    available_moves.append(("onBoard_MoveDice", i))
                    
        return np.array(available_moves)
 

    def canMoveToGoal(self, player_pieces, move_pieces, dice):
        available_moves = []
        for piece in move_pieces:
            if player_pieces[piece] + dice == 57:
                available_moves.append(("onBoard_GoGoal", piece))
        return np.array(available_moves)


    def canMoveToStar(self, player_pieces, move_pieces, enemy_pieces, dice):
        available_moves = []
        # for each moveable piece
        for piece in move_pieces:
            next_position = player_pieces[piece] + dice
            # if it will land on a star, given the eyes on the dice
            if self.willLandOnStar(next_position) and not self.pieceCanDie(piece=piece, player_pieces=player_pieces, enemy_pieces=enemy_pieces, dice=dice):
                available_moves.append(("onBoard_Star", piece))
        #print("------  canMoveToStar")
        return np.array(available_moves)


    def canMoveToGlobe(self, player_pieces, move_pieces, enemy_pieces, dice):
        available_moves = []
        # for each moveable piece
        for piece in move_pieces:
            next_position = player_pieces[piece] + dice
            # if it will land on a globe, given the eyes on the dice
            if self.willLandOnGlobe(next_position) and next_position not in enemy_pieces:
                available_moves.append(("onBoard_Globe", piece))
        #print("------  canMoveToGlobe")
        return np.array(available_moves)


    def canProtect(self, player_pieces, move_pieces, dice):
        available_moves = []
        # for each moveable piece
        for piece in move_pieces:
            next_position = player_pieces[piece] + dice
            # if it will land on a teammate, given the eyes on the dice (a funky way to avoid it seeing itself)
            if next_position in player_pieces[np.arange(len(player_pieces))!=piece] and next_position not in self.star_positions:
                available_moves.append(("onBoard_Protect", piece))
        #print("------  canProtect")
        return np.array(available_moves)


    def canDie(self, player_pieces, move_pieces, enemy_pieces, dice):
        available_moves = []
        # for every piece that can be moved
        for piece in move_pieces:
            if self.pieceCanDie(piece=piece, player_pieces=player_pieces, enemy_pieces=enemy_pieces, dice=dice):
                available_moves.append(("onBoard_Die", piece))
        #print("------  canDie")
        return np.array(available_moves)


    def canKill(self, player_pieces, move_pieces, enemy_pieces, dice):
        available_moves = []

        for piece in move_pieces:
            next_position = player_pieces[piece] + dice
            if not self.pieceCanDie(piece=piece, player_pieces=player_pieces, enemy_pieces=enemy_pieces, dice=dice) and next_position in enemy_pieces and player_pieces[piece] != 0 and next_position not in self.danger_positions:
                available_moves.append(("onBoard_Kill", piece))
        #print("------  canKill")
        return np.array(available_moves)


    def canMoveToGoalZone(self, player_pieces, move_pieces, enemy_pieces, dice):
        available_moves = []

        for piece in move_pieces:
            next_position = player_pieces[piece] + dice
            if next_position in self.goal_zone_positions and not self.pieceCanDie(piece=piece, player_pieces=player_pieces, enemy_pieces=enemy_pieces, dice=dice):
                available_moves.append(("onBoard_GoGoalZone", piece))
        #print("------  canMoveToGoalZone")
        return np.array(available_moves)


    def getAllAvailableMoves(self, player_pieces, move_pieces, enemy_pieces, dice):
        
        move_out = list(self.canMoveOut(player_pieces=player_pieces, move_pieces=move_pieces, dice=dice))
        move_dice = list(self.canMoveDice(player_pieces=player_pieces, move_pieces=move_pieces, enemy_pieces=enemy_pieces, dice=dice))
        move_goal = list(self.canMoveToGoal(player_pieces=player_pieces, move_pieces=move_pieces, dice=dice))
        move_star = list(self.canMoveToStar(player_pieces=player_pieces, move_pieces=move_pieces, enemy_pieces=enemy_pieces, dice=dice))
        move_globe = list(self.canMoveToGlobe(player_pieces=player_pieces, move_pieces=move_pieces, enemy_pieces=enemy_pieces, dice=dice))
        move_protect = list(self.canProtect(player_pieces=player_pieces, move_pieces=move_pieces, dice=dice))
        move_die = list(self.canDie(player_pieces=player_pieces, move_pieces=move_pieces, enemy_pieces=enemy_pieces, dice=dice))
        move_kill = list(self.canKill(player_pieces=player_pieces, move_pieces=move_pieces, enemy_pieces=enemy_pieces, dice=dice))
        move_goalzone = list(self.canMoveToGoalZone(player_pieces=player_pieces, move_pieces=move_pieces, enemy_pieces=enemy_pieces, dice=dice))

        res = move_out + move_dice + move_goal + move_star + move_globe + move_protect + move_die + move_kill + move_goalzone
        return np.array(res)


    def getBestMoves(self, available_moves, move_pieces):

        best = -10000
        action = ""
        for k, v in available_moves:
            if float(self.qtable.getValue(k)) > best:
                best = float(self.qtable.getValue(k))
                action = (k, int(v))
        
        return action
    

    def getBestMovesGeneric(self, available_moves, move_pieces):

        qtable = {
            'atSpawn_MoveOut' : 2,
            'atSpawn_MoveDiceValue' : -3,
            'atSpawn_Star' : -3,
            'atSpawn_Globe' : -3,
            'atSpawn_Protect' : -3,
            'atSpawn_Kill' : -3,
            'atSpawn_Die' : -3,
            'atSpawn_GoGoalZone' : -3,
            'atSpawn_GoGoal' : -3,
            
            'onBoard_MoveOut' : -3,
            'onBoard_MoveDice' : 0,
            'onBoard_Star' : 0.07,
            'onBoard_Globe' : 0.05,
            'onBoard_Protect' : 0.1,
            'onBoard_Kill' : 0.2,
            'onBoard_Die' : -0.2,
            'onBoard_GoGoalZone' : 0.19,
            'onBoard_GoGoal' : 0.4,
        }
        
        best = -10000
        action = ""
        for k, v in available_moves:
            if float(qtable[k]) > best:
                best = float(self.qtable.getValue(k))
                action = (k, int(v))
        
        return action


    def mapEnemyPositionRelativeToAI(self, enemy_pieces):
        mappedPositions = copy.copy(enemy_pieces)

        for i, player in enumerate(mappedPositions):
            for j, piece in enumerate(player):
                if piece == 0 or piece > 51:
                    mappedPositions[i][j] = 0
                else:
                    mappedPositions[i][j] = (piece + 13*(i+1))%52
        
        return mappedPositions


    def plotRewards(self, all_rewards, iterations, window_size):
        rewards = all_rewards

        X = [x for x in range(iterations-window_size)]

        rewards = movingAverage(rewards, window_size=window_size)

        plt.plot(X, rewards, color='g', label='rewards')
        plt.xlabel("Iterations")
        plt.ylabel("& and magnitude")
        plt.title("winrate and rewards")
        plt.legend()
        plt.show()


    def plotAllWinrates(self, winrates, iterations):
        colors = ["b", "g", "r", "m", "y", "k"]
        X = [x for x in range(iterations-1)]
        for i, wr in enumerate(winrates):
            plt.plot(X, wr, color=colors[i%6], label='fold ' + str(i))
            
        plt.xlabel("Iterations")
        plt.ylabel("& and magnitude")
        plt.title("winrate and rewards")
        plt.legend()
        plt.show()


    def reset(self):
        self.epsilon = self.e_greedy
        self.qtable.reset()


    def writeDataToCSV(self, data, name):

        # opening the csv file in 'a+' mode
        file = open(name, 'w', newline ='')

        # writing the data into the file
        with file:   
            write = csv.writer(file)
            write.writerows(data)


    def play(self, iterations):
        game = ludopy.Game(ghost_players=[2])
        there_is_a_winner = False

        player_win = 0

        for it in range(iterations):
            #print('Iteration: ', it)
            there_is_a_winner = False
            game.reset()

            while not there_is_a_winner:
                (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,there_is_a_winner), player_i = game.get_observation()
                if len(move_pieces):
                    if self.id == player_i:
                        #print("i move")   
                        mappedPositions = self.mapEnemyPositionRelativeToAI(enemy_pieces=enemy_pieces) 
                        available_moves = self.getAllAvailableMoves(player_pieces=player_pieces, move_pieces=move_pieces, enemy_pieces=mappedPositions, dice=dice)
                        state_and_action, piece_to_move = self.getBestMoves(available_moves=available_moves, move_pieces=move_pieces)
                    else:
                        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                else:
                    piece_to_move = -1
                _, _, _, _, playerIsAWinner, there_is_a_winner = game.answer_observation(piece_to_move)


                if there_is_a_winner:
                    if game.first_winner_was == self.id:
                        #print("i win")
                        player_win += 1
                    else:
                        pass

        return player_win

        
    def calcRegret(self, iteration, sum_reward, sum_best_reward):
        regret = (1/iteration)*sum_best_reward - (1/iteration)*sum_reward
        return regret

        

    def learn(self, iterations, savefile_winrate, savefile_reward, num_of_players = 4, folds = 10):

        myAIwon = 0
        # initialize a ludo game with specified amount of players (-1 since the last is our ai)
        # the numebr prevents the x player from moving out of spawn
        if num_of_players == 4:
            game = ludopy.Game(ghost_players=[])
        elif num_of_players == 3:
            game = ludopy.Game(ghost_players=[1])
        else:
            game = ludopy.Game(ghost_players=[1,3])

        final_winrates_all_folds = []
        winrates_for_all_folds = []
        rewards_for_all_folds = []
        regrets = []
        total_regret = 0

        # nr of folds
        for fold in range(folds):
            myAIwon = 0
            all_winrates = []
            all_rewards = []
            self.reset()
            game.reset()

            # nr of games played in each fold
            for i in range(1,iterations):

                if not (i%100):
                    print(f"fold {fold} iteration {i}")
                there_is_a_winner = False

                # new game for each iteration
                game.reset()

                # some initial values
                old_max_q_val = "onBoard_MoveDice"
                reward = 0
                best_reward = 0
                total_rewards = 0
                
                # play until someone wins
                while not there_is_a_winner:
                    (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = game.get_observation()

                    #print(f"player id: {player_i}  can move; {len(move_pieces) != 0}")
                    if len(move_pieces):
                        # if it is the AI's time to move
                        if self.id == player_i:    
                            # mapping enemy positions relative to the AI 
                            mappedPositions = self.mapEnemyPositionRelativeToAI(enemy_pieces=enemy_pieces) 

                            # finding all available moves
                            available_moves = self.getAllAvailableMoves(player_pieces=player_pieces, move_pieces=move_pieces, enemy_pieces=mappedPositions, dice=dice)
                            the_best, _ = self.getBestMovesGeneric(available_moves=available_moves, move_pieces=move_pieces)
                            # can take a random action. Depends on magnitude of epsilon
                            if np.random.uniform(0,1) < self.epsilon:
                                state_and_action, piece_to_move = random.choice(available_moves)
                                piece_to_move = int(piece_to_move)

                                # next_best, _ = self.getBestMoves(available_moves=available_moves, move_pieces=move_pieces)
                                # self.qtable.updateQtable(key=old_max_q_val, reward=reward, next_best=next_best, gamma=self.gamma, learning_rate=self.lr)
                                # old_max_q_val = state_and_action

                                self.qtable.updateQtable(key=old_max_q_val, reward=reward, next_best=state_and_action, gamma=self.gamma, learning_rate=self.lr)
                                old_max_q_val = state_and_action
                            else:
                                state_and_action, piece_to_move = self.getBestMoves(available_moves=available_moves, move_pieces=move_pieces)
                                self.qtable.updateQtable(key=old_max_q_val, reward=reward, next_best=state_and_action, gamma=self.gamma, learning_rate=self.lr)
                                old_max_q_val = state_and_action

                            # give reward 
                            reward = self.rewards.getReward(state_and_action)
                            best_reward += self.rewards.getReward(the_best)
                            total_rewards += reward

                        else:
                            # if it is one of the other players, just do a random move
                            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                    else:
                        piece_to_move = -1
                    _, _, _, _, playerIsAWinner, there_is_a_winner = game.answer_observation(piece_to_move)

                total_regret += self.calcRegret(iteration=i, sum_reward=total_rewards, sum_best_reward=best_reward)
                regrets.append(self.calcRegret(iteration=i, sum_reward=total_rewards, sum_best_reward=best_reward)+total_regret)

                # save winrate after each game played
                all_winrates.append((myAIwon/i)*100)
                all_rewards.append(total_rewards)
                #print(f"Done with iterations {i} in fold: {fold}")
                #if i != 0:
                #    print(f"winrate: {int((myAIwon/i)*100)}")

                # epsilon decay. Will make it exploit more after each game, instead of exploring
                self.epsilon = max(self.min_epsilon, np.exp(-self.epsilon_decay*i))
                    
                if game.first_winner_was == self.id:
                    myAIwon += 1

            y = [i for i, _ in enumerate(regrets)]
            plt.plot(y, regrets, color='g', label='rewards')
            plt.xlabel("Iterations")
            plt.ylabel("& and magnitude")
            plt.title("winrate and rewards")
            plt.legend()
            plt.show()

            # file = open('csv_files/regrets/decay1_0_01.csv', 'w', newline ='')

            # # writing the data into the file
            # with file: 
            #     write = csv.writer(file)
            #     write.writerow(regrets)

            # qtable_ = self.getQtable()
            # with open('csv_files/qtables/fold'+str(fold)+'.csv', 'w') as f:
            #     for key in qtable_.keys():
            #         f.write("%s,%s\n"%(key,qtable_[key]))

            # save all winrates for all folds
            winrates_for_all_folds.append(all_winrates)
            rewards_for_all_folds.append(all_rewards)
            #rewards_for_all_folds.append(movingAverage(values=all_rewards, window_size=10))
            final_winrates_all_folds.append((myAIwon/iterations)*100)
            print(f"fold: {fold} won {myAIwon} times or {(myAIwon/iterations)*100}% of the games")


        #self.plotRewards(all_rewards=all_rewards, iterations=iterations, window_size=50)
        average = 0
        for x in final_winrates_all_folds:
            average += x
        print(f"Average winrate: {average/folds}  with learning rate: {self.lr} and reward decay: {self.gamma}")
        #self.writeDataToCSV(data=winrates_for_all_folds, name=savefile_winrate)
        #self.writeDataToCSV(data=rewards_for_all_folds, name=savefile_reward)

        #self.plotAllWinrates(winrates=winrates_for_all_folds, iterations=iterations)
