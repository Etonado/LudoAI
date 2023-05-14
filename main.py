import ludopy
import numpy as np
from genetic_learning import GeneticPlayer
import input_generator
import matplotlib.pyplot as plt

TRAIN = True

if TRAIN:
    # initialize genetic player
    genetic_player = GeneticPlayer()

    genetic_player.train()


else:
    # play with trained player
    g = ludopy.Game()
    while True:
        g.reset()
        there_is_a_winner = False
        while not there_is_a_winner:
            
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()
            '''
            print("Player: ", player_i,end=" ")
            print("Move Pieces: ",move_pieces,end=" ")
            print("Player Pieces ",player_pieces,end=" ")
            print("enemy_pieces: ",enemy_pieces)

            '''
            
            if player_i == 0:
                # player to be trained
                # set mask for all indices set in move_pieces
                mask = np.zeros(4)
                for x in move_pieces:
                    mask[x] = 1

                #print("\nPieces: ",g.get_pieces()[0])

                # decision based on player state        
                if len(move_pieces)>1:
                    # obtain the current sate of the pieces
                    pieces = g.get_pieces()
                    # obtain current state
                    input_matrix = input_generator.generate_inputs(player_i,pieces,mask,dice)

                    #piece_to_move = function call
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))] # to be replaced by function call

                elif len(move_pieces) == 1: # there is no choice for moving a piece
                    piece_to_move = move_pieces[0]
                else:   # no piece can be moved
                    piece_to_move = -1

            else:
                # other players
                if len(move_pieces):
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                else:
                    piece_to_move = -1


            _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)


        print("Player: ", player_i," won")


        '''
        print("Saving history to numpy file")
        g.save_hist("./history/game_history.npy")
        print("Saving game video")
        g.save_hist_video("./history/game_video.mp4")
        '''