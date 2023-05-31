import ludopy
import numpy as np
from genetic_learning import GeneticPlayer, InputGenerator
import matplotlib.pyplot as plt
import Ditlev.agent as agent
import Ditlev.multiplay as multiplay
from Ditlev.multiplay import chooseAction
import csv




def determine_number_of_games(winning_rate_list, expectation, range,show = False):
    if show:
        plt.figure()
        plt.plot(winning_rate_list)
        plt.ylabel('Winning rate')
        plt.xlabel('Number of games')
        axis = plt.gca()
        #axis.set_ylim([0.2,0.3])
        #axis.set_xlim([0,1000])
        plt.show()

    highest_bad_approximation = 0

    for idx,elements in enumerate(winning_rate):
        if elements < (expectation - range) or elements > (expectation + range):
           highest_bad_approximation = idx

    print("Highest bad approximation: ",highest_bad_approximation)

TRAIN = False
PLAY = False
EVALUATE = False
WEIGHTS = False



# initialize genetic player
genetic_player = GeneticPlayer()
input_generator = InputGenerator()



if TRAIN:
    # train genetic player
    genetic_player.train()

elif EVALUATE:
    # play with best member of population
    genetic_player.evaluate_best(number_of_games = 1000)



elif PLAY:
    # play with trained player
    winning_rate = []
    g = ludopy.Game()
    for i in range(50):
        games_played = 0
        games_won = 0
        while games_played < 300:
            g.reset()
            there_is_a_winner = False
            games_played += 1
            while not there_is_a_winner:
                
                (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()


                if player_i == 0:
                   piece_to_move = genetic_player.play_with_best(player_i,move_pieces,g.get_pieces(),dice)

 
                else:
                    # other players
                    if len(move_pieces):
                        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                    else:
                        piece_to_move = -1


                _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

            if player_i == 0:
                games_won += 1

        winning_rate.append(games_won/games_played)
    print(winning_rate)
    with open('./6n_rate.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(winning_rate)


            
elif WEIGHTS:
    weights = genetic_player.get_weights_of_best_player()
    weights = np.array(weights).astype(float)
    weights = np.reshape(weights,(10,-1))

    with open('./weights_best_6.csv', 'w') as file:
        writer = csv.writer(file)
        for lines in weights:
            writer.writerow(lines)

else:
    print("Please choose an action to be performed by setting the variables TRAIN, PLAY, EVALUATE or WEIGHTS to True")