import os
import sys
import ludopy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from Ditlev.agent import Agent


def chooseAction(agent : Agent, enemy_pieces, player_pieces, move_pieces, dice):
    mappedPositions = agent.mapEnemyPositionRelativeToAI(enemy_pieces=enemy_pieces) 
    available_moves = agent.getAllAvailableMoves(player_pieces=player_pieces, move_pieces=move_pieces, enemy_pieces=mappedPositions, dice=dice)
    state_and_action, piece_to_move = agent.getBestMoves(available_moves=available_moves, move_pieces=move_pieces)

    # give reward 
    reward = agent.rewards.getReward(state_and_action)
    #total_rewards += reward

    return piece_to_move


def playLudo(player0 : Agent, player1 : Agent, player2 : Agent, player3: Agent, iterations):
    game = ludopy.Game(ghost_players=[2, 3])
    there_is_a_winner = False
    reward = 0
    total_rewards = 0

    player1_win = 0
    player2_win = 0
    player3_win = 0
    player4_win = 0

    for it in range(iterations):
        if not it%100:
            print('Iteration: ', it)
        there_is_a_winner = False
        game.reset()

        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,there_is_a_winner), player_i = game.get_observation()
            if len(move_pieces):
                if player0.id == player_i:   
                    piece_to_move = chooseAction(player0, enemy_pieces=enemy_pieces, player_pieces=player_pieces, move_pieces=move_pieces, dice=dice)
                elif player1.id == player_i:
                    piece_to_move = chooseAction(player1, enemy_pieces=enemy_pieces, player_pieces=player_pieces, move_pieces=move_pieces, dice=dice)
                else:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                # elif player2.id == player_i:
                #     piece_to_move = chooseAction(player2, enemy_pieces=enemy_pieces, player_pieces=player_pieces, move_pieces=move_pieces, dice=dice)
                # else:
                #     piece_to_move = chooseAction(player3, enemy_pieces=enemy_pieces, player_pieces=player_pieces, move_pieces=move_pieces, dice=dice)
            else:
                piece_to_move = -1
            _, _, _, _, playerIsAWinner, there_is_a_winner = game.answer_observation(piece_to_move)


            if there_is_a_winner:
                if game.first_winner_was == player0.id:
                    player1_win += 1
                    # print("player0 won")
                elif game.first_winner_was == player1.id:
                    player2_win += 1
                    # print("player1 won")
                elif game.first_winner_was == player2.id:
                    player3_win += 1
                    # print("player2 won")
                elif game.first_winner_was == player3.id:
                    player4_win += 1
                    # print("player3 won")
                else:
                    print("den skal ikke herned")
    
    return player1_win, player2_win, player3_win, player4_win


def plotwins(first, second, third, fourth, iterations):

    labels = 'Player 0 (Q)', 'Player 1 (SARSA)'#, 'Player 2 (fill)', 'Player 3 (fill2)'
    sizes = [(first/iterations)*100, (second/iterations)*100]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', explode=(0, 0), shadow = False)
    plt.show()