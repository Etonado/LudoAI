import numpy as np

SINGLE_PLAYER_POSITION_OFFSET = 13


class LudoState():
    def __init__(self,globes=True,stars=True):
        self.globes_enabled = globes
        self.stars_enabled = stars
    

def generate_inputs(player_i,pieces,active_player_mask, dice_roll:int):
    
    opponent_players = __get_diff_position(player_i,pieces,active_player_mask)
    change_danger = __get_change_danger(opponent_players,dice_roll)
    goal_position = __get_goal_position(player_i,pieces,active_player_mask)   
    globe_position = __get_special_position(player_i,pieces,active_player_mask,"globe")
    star_position = __get_special_position(player_i,pieces,active_player_mask,"star")

    # input 0: normalized dice roll
    i_0 = dice_roll *np.ones(4)/6

    # input 1: can kick out one player
    i_1 = np.zeros(4)
    
    for idx,player in enumerate(opponent_players):
        for distances in player:
            if distances == dice_roll:
                i_1[idx] = 1  
        
    # input_2: changes danger by moving

    # positive: danger increases
    # negative: danger decreases
    # 0: no change
    # danger = number of opponents that can kick out player
    i_2 = change_danger
    
    # input 3: can reach home
    i_3 = np.zeros(4)
    for idx,distance in enumerate(goal_position):
        if distance < dice_roll and distance != 0:
            i_3[idx] = 1

    # input 4: can reach safe zone
    i_4 = np.zeros(4)
    for idx,distance in enumerate(goal_position):
        if distance-7 < dice_roll and distance-7 >= 0:
            i_4[idx] = 1

    # input 5: can reach free globe // # input 6: can reach occupied globe
    i_5 = np.zeros(4)
    i_6 = np.zeros(4)
    reach_globe = np.zeros(4)

    for idx,distance in enumerate(globe_position):
        for entries in  distance:
            if entries == dice_roll:
                reach_globe[idx] = 1
    
    i_6 = ((reach_globe == 1) & (i_1 == 1)).astype(float)
    i_5 = ((reach_globe == 1) & (i_1 != 1)).astype(float)


    # update i_1 for occupied globes
    i_1 = ((reach_globe != 1) & (i_1 == 1)).astype(float)


    # input 7: can reach star
    i_7 = np.zeros(4)
    for idx,distance in enumerate(star_position):
        for entries in  distance:
            if entries == dice_roll:
                i_7[idx] = 1

    # input 8: normalized distance to goal
    i_8 = goal_position/59

    I = np.vstack((i_0,i_1,i_2,i_3,i_4,i_5,i_6,i_7,i_8))
    return I

def __get_diff_position(player_i,pieces,mask):
    # distance from one starting point to the next one
    global SINGLE_PLAYER_POSITION_OFFSET
    # extract pieces from list
    pieces = pieces[0]
    # get pieces of current player
    my_pieces = pieces[player_i]
    # Disregard pieces that are save
    mask = np.logical_and(mask,my_pieces <(4*SINGLE_PLAYER_POSITION_OFFSET))
    # List of opponent pieces
    all_opponent_pieces = []
    # Create list with entries of difference positions for all active pieces
    player_rel_positions = np.array([[],[],[],[]])
    # set array elements to be a list
    player_rel_positions = player_rel_positions.tolist()
    
    for i in range(4):
        # Skip the pieces from the current player
        if i == player_i:
            continue
        # Get pieces of ith opponent player
        opponent_player = pieces[i]
        # Remove all zero entries
        opponent_player = opponent_player[opponent_player != 0]
        # Remove save pieces
        opponent_player = opponent_player[opponent_player <= (4*SINGLE_PLAYER_POSITION_OFFSET+1)]
        # Project all positions to starting point of current player
        player_offset = (i-player_i) * SINGLE_PLAYER_POSITION_OFFSET
        opponent_player = [x + player_offset for x in opponent_player]

        # Subtract 4*player_offset from all entries that are greater than 52 to account for players that have passed the starting point of current player
        for idx,x in enumerate(opponent_player):
            if x > 52:
                opponent_player[idx] = x - 52

        # Add single positions to list
        for x in opponent_player:
            all_opponent_pieces.append(x)
        
    # Players in all_opponent_pieces-list are now in the same coordinate system
    # Add relative positions to list
    for idx,list in enumerate(player_rel_positions):
        if(mask[idx]==True):
            for x in all_opponent_pieces:
                list.append(x-my_pieces[idx])


    # Players that are more than half the fields away are seen from the other side
    for player in player_rel_positions:
        for idx,x in enumerate(player):
            if x > 26:
                player[idx] = x - 52
        '''
            Theoretically neccessary but not needed because players with negative distances have passed the goal position and can thus not be reached by the current player
            elif x < -26:
                player[idx] = x + 52
        '''
    # Narrow the range to [-6 and 6], other entries are discarded
    for i in range(4):
        player_rel_positions[i] = [x for x in player_rel_positions[i] if x >= -6 and x <= 6]
       

    return player_rel_positions


def __get_special_position(player_i,pieces,mask,type:str):

    if type == "globe":
        pos1 = 8
        pos2 = 13
    elif type == "star":
        pos1 = 4
        pos2 = 11
    else:
        raise ValueError("Invalid type: " + type)

    global SINGLE_PLAYER_POSITION_OFFSET
    # extract pieces from list
    pieces = pieces[0]
    # get pieces of current player
    my_pieces = pieces[player_i]
    # Disregard pieces that are save
    mask = np.logical_and(mask,my_pieces <(4*SINGLE_PLAYER_POSITION_OFFSET))

    # Create list with entries of difference positions to globes
    special_rel_positions = np.array([[],[],[],[]])
    # set array elements to be a list
    special_rel_positions = special_rel_positions.tolist()
    # Project all positions in one quarter of the board
    my_pieces = [x % SINGLE_PLAYER_POSITION_OFFSET for x in my_pieces]
    # Get distances for all active pieces
    for idx,lists in enumerate(special_rel_positions):
        if(mask[idx]==True):
            lists.append(pos1-my_pieces[idx])
            lists.append(pos2-my_pieces[idx])

    
    for i in range(4):
        special_rel_positions[i] = [x for x in special_rel_positions[i] if x >= -6 and x <= 6]

    return special_rel_positions

def __get_goal_position(player_i,pieces,mask):
    # extract pieces from list
    pieces = pieces[0]
    # get pieces of current player
    my_pieces = pieces[player_i]
    # Disregard pieces that are save
    goal_distance = [59 - x for x in my_pieces]
    goal_distance = np.multiply(goal_distance,mask)
    return goal_distance

def __get_change_danger(opponent_players,dice_roll):
    # Initialize danger arrays
    current_danger = np.zeros(4)
    future_danger = np.zeros(4)
    # Add danger for current position
    for idx,player in enumerate(opponent_players):
        for distances in player:
            # distances between -1 and -6 might be dangerous for the current player
            if distances <= -1 and distances >= -6:
                current_danger[idx] += 1
            
    # Add danger for future position
    for idx,player in enumerate(opponent_players):
        for distances in player:
            # distances between -1 and -6 might be dangerous for the current player
            if distances - dice_roll <= -1 and distances - dice_roll >= -6:
                future_danger[idx] += 1

    return (future_danger-current_danger)

