from Ditlev.agent import Agent
import ludopy
import numpy as np
from Ditlev.multiplay import chooseAction
from genetic_learning import GeneticPlayer, InputGenerator
import csv


# initialize genetic player
genetic_player = GeneticPlayer()
input_generator = InputGenerator()

def playLudo(player0 : Agent, iterations):
    game = ludopy.Game()
    there_is_a_winner = False
    reward = 0
    total_rewards = 0

    player1_win = 0
    player2_win = 0
    player3_win = 0
    player4_win = 0

    for it in range(iterations):
        there_is_a_winner = False
        game.reset()

        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,there_is_a_winner), player_i = game.get_observation()
            if len(move_pieces):
                if player0.id == player_i:   
                    piece_to_move = chooseAction(player0, enemy_pieces=enemy_pieces, player_pieces=player_pieces, move_pieces=move_pieces, dice=dice)
                elif 1 == player_i:
                    piece_to_move = genetic_player.play_with_best(player_i,move_pieces,game.get_pieces(),dice)
                else:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1
            _, _, _, _, playerIsAWinner, there_is_a_winner = game.answer_observation(piece_to_move)


            
            if there_is_a_winner:
                if game.first_winner_was == player0.id:
                    player1_win += 1
                    # print("player0 won")
                elif game.first_winner_was == 1:
                    player2_win += 1
                    # print("player1 won")
                elif game.first_winner_was == 2:
                    player3_win += 1
                #     # print("player2 won")
                elif game.first_winner_was == 3:
                    player4_win += 1
                    # print("player3 won")
                else:
                    print(game.first_winner_was)
                    print("den skal ikke herned")
    
    return player1_win, player2_win, player3_win, player4_win


if __name__ == "__main__":
    player_1 = Agent(id=0)
    player_1.setQtable("./Ditlev/fold6_Q.csv")
    print(player_1.qtable)

    win_d = []
    win_e = []
    win_r = []
    for i in range(50):
        win1,win2,win3,win4 = playLudo(player_1, 300)
        win_d.append(win1)
        win_e.append(win2)
        win_r.append(win3)
        print("Iteration: ", i)
    
    with open('./d_rate.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(win_d)
    
    with open('./e_rate.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(win_e)

    with open('./r_rate.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(win_r)

