import numpy as np
import ludopy
from multiprocessing import Pool
import time



NUMBER_OF_HIDDEN_NEURONS = 6
NUMBER_OF_INPUTS = 9
BIT_LENGTH = 8
NUMBER_OF_WEIGHTS = NUMBER_OF_INPUTS * NUMBER_OF_HIDDEN_NEURONS + NUMBER_OF_HIDDEN_NEURONS
GENE_LENGTH = BIT_LENGTH * NUMBER_OF_WEIGHTS
NUMBER_OF_CORES = 4 # can't be changed atm :/
SINGLE_PLAYER_POSITION_OFFSET = 13 # in game field number



class GeneticPlayer:

    global GENE_LENGTH, NUMBER_OF_CORES

    def __init__(self,population_size = 100, fitness_loops = 300, mutation_rate = 0.1, elite_rate = 0.1, parent_rate = 0.3, selection_type = "tournament", multi_core_learning = True):
        self.variants = []
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.multi_core_learning = multi_core_learning
        self.generation = 0
        self.elite_size = int(elite_rate*population_size)
        self.parent_size  = int(parent_rate*population_size)
        self.selection_type = selection_type
        self.tournament_size = 4
        self.fitness_loops = fitness_loops
        self.game = ludopy.Game()
        self.input_generator = InputGenerator()
        self.__generate_initial_population()

    
    def __generate_initial_population(self):
        for i in range(self.population_size):
            self.variants.append(Chromosome(np.random.randint(0, 2, size=GENE_LENGTH)))

    def __sort_population(self):
        self.variants.sort(key=lambda x: x.fitness, reverse=True)

    def compute_fitness(self,process_idx,random_seed = None,players="random"):
        # fitness for all chromosomes in the current generation
        chromosome_fitness = []
        # set random seed if given
        if random_seed:
            np.random.seed(random_seed)   
            
        max_fitness = 0
        for chromosome in self.variants[process_idx[0]:process_idx[1]]:
            # every chromosome is tested for fitness_loops times
            weights = chromosome.decrypt_chromosome()
            winning_count = 0
            for i in range(self.fitness_loops):
                # reset the game
                self.game.reset()
                # reset winner state
                there_is_a_winner = False
                # play the game until there is a winner
                while not there_is_a_winner:
                    # one game is played
                    # reset the piece to move
                    player_with_highest_activation = -1
                    highest_activation = -100
                    # get the current state of the game
                    (dice, move_pieces, _, _, _, there_is_a_winner), player_i = self.game.get_observation()
                    # player to be trained
                    if player_i == 0:
                        # set mask for all indices set in move_pieces
                        mask = np.zeros(4)
                        for x in move_pieces:
                            mask[x] = 1
                        # decision based on player state        
                        if len(move_pieces)>1:
                            pieces = self.game.get_pieces()
                            # generate input matrix from the current state
                            I = self.input_generator.generate_inputs(player_i,pieces,mask,dice)
                            for idx,pieces in enumerate(move_pieces):
                                input = I[:,pieces]
                                activation = self.run_neural_networks(input,weights)
                                if activation > highest_activation:
                                    highest_activation = activation
                                    player_with_highest_activation = idx

                            #piece_to_move = function call
                            if player_with_highest_activation == -1:
                                # something went wrong
                                print("Something went wrong with the neural network")
                                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                            else:
                                # move the piece with the highest activation
                                piece_to_move = move_pieces[player_with_highest_activation] 

                        elif len(move_pieces) == 1: # there is no choice for moving a piece
                            piece_to_move = move_pieces[0]
                        else:   # no piece can be moved
                            piece_to_move = -1

                    elif players == "random":
                        # other players
                        if len(move_pieces):
                            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                        else:
                            piece_to_move = -1
                    else:
                        raise Exception("Invalid player type")

                    # get response from the game
                    _, _, _, _, _, there_is_a_winner = self.game.answer_observation(piece_to_move)

                if player_i == 0:
                    winning_count += 1
                #print("Player: ", player_i," won")
            
        
            # fitness_loop amount of games have been played
            chromosome.fitness = winning_count/self.fitness_loops
            chromosome_fitness.append(chromosome.fitness)

            if chromosome.fitness > max_fitness:
                max_fitness = chromosome.fitness
            print("Fitness: ", chromosome.fitness," Generation: ", self.generation, " Max fitness: ",max_fitness)

        if self.multi_core_learning:
            return chromosome_fitness



    def run_neural_networks(self,input_matrix,weights,activation_function = "ReLU"):
        global NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_INPUTS

        # reshape the input matrix to a 1 x Inputs matrix
        input_matrix = np.reshape(input_matrix,(1,NUMBER_OF_INPUTS))
        # reshape the weights to a Inputs x Neurons matrix
        input_weights = np.reshape(weights[0:(NUMBER_OF_HIDDEN_NEURONS*NUMBER_OF_INPUTS)],(NUMBER_OF_INPUTS,NUMBER_OF_HIDDEN_NEURONS))
        # reshape the weights to a Neurons x 1 matrix
        hidden_weights = np.reshape(weights[NUMBER_OF_HIDDEN_NEURONS*NUMBER_OF_INPUTS:],(NUMBER_OF_HIDDEN_NEURONS,1))

        # calculate the activation of the hidden layer
        hidden_activation = np.matmul(input_matrix,input_weights)
        # apply the activation function
        if activation_function == "ReLU":
            hidden_activation = np.maximum(hidden_activation,0)
        elif activation_function == "sigmoid":
            hidden_activation = 1/(1+np.exp(-hidden_activation))
        else:
            raise Exception("Invalid activation function")
            
        
        # calculate the output
        output = np.matmul(hidden_activation,hidden_weights)

        return output

    def __tournament_selection(self,k):
        participants = []
        # select k random chromosomes
        for i in range(k):
            participants.append(self.variants[np.random.randint(0, self.population_size)])

        # return the chromosome with the highest fitness
        participants.sort(key=lambda x: x.fitness, reverse=True)
        return participants[0]

    
        
    def __select_parents(self,selection_type):
        # Create two lists of parents
        parents1 = []
        parents2 = []
        # Select parents
        if selection_type == "tournament":
            for i in range(self.parent_size):
                parents1.append(self.__tournament_selection(self.tournament_size))
                parents2.append(self.__tournament_selection(self.tournament_size))
            return parents1,parents2

        
        elif selection_type == "roulette_wheel":
            self.__roulette_wheel_selection()   
            return parents1,parents2
        else:
            raise Exception("Invalid selection type") 
        
    def __average_fitness(self):
        sum = 0
        for chromosome in self.variants:
            sum += chromosome.fitness
        return sum/self.population_size
    
    def __mean_std_fitness(self):
        fitness = []
        for chromosome in self.variants:
            fitness.append(chromosome.fitness)
        
        mean = np.mean(fitness)
        std = np.std(fitness)

        return mean,std


    def __call_fitness_function(self):
        if self.multi_core_learning:
            # initialize random seeds
            random_seed_0 = np.random.randint(0,10000000)
            random_seed_1 = np.random.randint(0,10000000)
            random_seed_2 = np.random.randint(0,10000000)
            random_seed_3 = np.random.randint(0,10000000)
            # split variants in 4 groups
            range_0 = (0,int(self.population_size/4))
            range_1 = (int(self.population_size/4),int(self.population_size/2))
            range_2 = (int(self.population_size/2),int(3*self.population_size/4))
            range_3 = (int(3*self.population_size/4),self.population_size)
            # create processes
            # def compute_fitness(self,process_idx = None,random_seed = None,players="random"):
            with Pool(processes=NUMBER_OF_CORES) as pool:
                inputs = [(range_0,random_seed_0,"random"), (range_1,random_seed_1,"random"),(range_2,random_seed_2,"random"),(range_3,random_seed_3,"random")]
                chromosome_fitness_lists = pool.starmap(self.compute_fitness, inputs)
                # flatten the list
                chromosome_fitness = [item for sublist in chromosome_fitness_lists for item in sublist]
                # set fitness of chromosomes
                for idx,chromosomes in enumerate(self.variants):
                    chromosomes.fitness = chromosome_fitness[idx]
        else:
            range = (0,self.population_size)
            self.compute_fitness(range)

    def train(self,generations = 50):
        # compute fitness of initial population
        time_start = time.time()
        self.__call_fitness_function()
        time_end = time.time()
        print("Time to compute fitness of initial population: ", time_end-time_start)
        self.generation = 1
        while self.generation < generations:
            # sort population
            self.__sort_population()
            # print fitness statistics of previous generation
            # print max fitness of generation
            print("Generation: ", self.generation -1, " Max fitness: ", self.variants[0].fitness)
            # print lowest fitness of generation
            print("Generation: ", self.generation -1, " Min fitness: ", self.variants[self.population_size-1].fitness)
            # print average fitness of generation
            print("Mean and std of fitness: ", self.__mean_std_fitness())
            mean,std = self.__mean_std_fitness()
            # save statistics
            with open('./statistics/file3.txt', 'a') as file:
                file.write("%d,%.3f,%.3f,%.3f,%.3f\n" % (self.generation-1,self.variants[self.population_size-1].fitness,self.variants[0].fitness,mean,std))

            best_5 = []
            for chromosome in self.variants[0:5]:
                best_5.append(chromosome.genes)
            best_5 = np.array(best_5)
            np.savetxt("./weights/best_5_current_gen.txt",best_5, delimiter=',',fmt='%1.4f',newline="\n")

            # select parents
            parents1,parents2 = self.__select_parents(self.selection_type)
            # extract elite chromosomes and delete rest of population
            self.variants = self.variants[0:self.elite_size]
            # create a list of children
            children = []
            # determine crossover method 
            number_single_point_crossover = np.random.randint(1,(self.population_size-self.elite_size))
            # single point crossover
            for i in range(number_single_point_crossover):
                children.append(parents1[np.random.randint(0,self.parent_size)].single_point_crossover(parents2[np.random.randint(0,self.parent_size)]))
            # two point crossover
            for i in range(self.population_size-self.elite_size-number_single_point_crossover):
                children.append(parents1[np.random.randint(0,self.parent_size)].two_point_crossover(parents2[np.random.randint(0,self.parent_size)]))
            # mutation
            for child in children:
                child.mutate(self.mutation_rate)
            # add children to population
            self.variants.extend(children)
            # compute fitness of new population
            time_start = time.time()
            self.__call_fitness_function()
            time_end = time.time()
            print("Time to compute fitness: ", time_end-time_start)
            # increase generation
            # print average fitness of generation

            self.generation += 1


    def get_weights_of_best_player(self):
        # open textfile to get genes of best player
        try:
            with open('./weights/best_current_gen.txt', 'r') as file:
                genes = np.loadtxt(file, delimiter=',', max_rows=1).astype(int)
        except:
            raise Exception("No best_current_gen.txt file found. Please evaluate the final players first.")
        
                
        winner = Chromosome(genes)
       
        weights = winner.decrypt_chromosome()

        return weights


    def evaluate_best(self,number_of_games,players = "random",plot = False):
        
        best_winning_rate = 0
        best_winning_rate_index = -1
        try:
            with open('./weights/best_5_current_gen.txt', 'r') as file:
                best_genes = np.loadtxt(file, delimiter=',', max_rows=5).astype(int)
        except:
            raise Exception("No best_5_current_gen.txt file found. Please train the model first.")
        
        best_variants = []
        for genes in best_genes:
            best_variants.append(Chromosome(genes))

        

        for player_index,chromosome in enumerate(best_variants):
            # every chromosome is tested for fitness_loops times
            weights = chromosome.decrypt_chromosome()

            winning_count = 0
            games_played = 0
            winning_rate_over_time = []

            while games_played < number_of_games:
                self.game.reset()
                there_is_a_winner = False
                games_played += 1

                # reset the game
                self.game.reset()
                # reset winner state
                there_is_a_winner = False
                # play the game until there is a winner
                while not there_is_a_winner:
                    # one game is played
                    # reset the piece to move
                    player_with_highest_activation = -1
                    highest_activation = -100
                    # get the current state of the game
                    (dice, move_pieces, _, _, _, there_is_a_winner), player_i = self.game.get_observation()
                    # player to be trained
                    if player_i == 0:
                        # set mask for all indices set in move_pieces
                        mask = np.zeros(4)
                        for x in move_pieces:
                            mask[x] = 1
                        # decision based on player state        
                        if len(move_pieces)>1:
                            pieces = self.game.get_pieces()
                            # generate input matrix from the current state
                            I = self.input_generator.generate_inputs(player_i,pieces,mask,dice)
                            for idx,pieces in enumerate(move_pieces):
                                input = I[:,pieces]
                                activation = self.run_neural_networks(input,weights)
                                if activation > highest_activation:
                                    highest_activation = activation
                                    player_with_highest_activation = idx

                            #piece_to_move = function call
                            if player_with_highest_activation == -1:
                                # something went wrong
                                print("Something went wrong with the neural network")
                                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                            else:
                                # move the piece with the highest activation
                                piece_to_move = move_pieces[player_with_highest_activation] 

                        elif len(move_pieces) == 1: # there is no choice for moving a piece
                            piece_to_move = move_pieces[0]
                        else:   # no piece can be moved
                            piece_to_move = -1

                    elif players == "random":
                        # other players
                        if len(move_pieces):
                            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                        else:
                            piece_to_move = -1
                    else:
                        raise Exception("Invalid player type")

                    # get response from the game
                    _, _, _, _, _, there_is_a_winner = self.game.answer_observation(piece_to_move)

                if player_i == 0:
                    winning_count +=1
                
                winning_rate_over_time.append(winning_count/games_played)

            final_winning_rate = winning_count/games_played
            print(int(final_winning_rate))
            print("Player with genes ",player_index," has a winning rate of: ",final_winning_rate)

            if final_winning_rate > best_winning_rate:
                best_winning_rate_index = player_index
                best_winning_rate = final_winning_rate

        best = []
        best.append(best_variants[best_winning_rate_index].genes)
        best = np.array(best)
        np.savetxt("./weights/best_current_gen.txt",best, delimiter=',',fmt='%1.4f',newline="\n")

  

    def play_with_best(self,player_i,move_pieces,pieces,dice):
        # init
        player_with_highest_activation = -1
        highest_activation = -1000

        mask = np.zeros(4)
        for x in move_pieces:
            mask[x] = 1


        if len(move_pieces)>1:
            # obtain the current sate of the pieces
            # obtain current state

            I = self.input_generator.generate_inputs(player_i,pieces,mask,dice)
            weights = self.get_weights_of_best_player()

            for idx,pieces in enumerate(move_pieces):
                input = I[:,pieces]
                activation = self.run_neural_networks(input,weights)
                if activation > highest_activation:
                    highest_activation = activation
                    player_with_highest_activation = idx


            #piece_to_move = function call
            if player_with_highest_activation == -1:
                # something went wrong
                print("Something went wrong with the neural network")
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                # move the piece with the highest activation
                piece_to_move = move_pieces[player_with_highest_activation] 

        elif len(move_pieces) == 1: # there is no choice for moving a piece
            piece_to_move = move_pieces[0]
        else:   # no piece can be moved
            piece_to_move = -1

        return piece_to_move



class Chromosome():
    def __init__(self,genes = []):
        self.genes = genes
        self.fitness = 0    

    def copy(self):
        return Chromosome(self.genes.copy())
    
    def single_point_crossover(self,other_chromosome):
        # single point crossover
        crossover_point = np.random.randint(0,GENE_LENGTH)
        new_genes = []
        for i in range(GENE_LENGTH):
            if i < crossover_point:
                new_genes.append(self.genes[i])
            else:
                new_genes.append(other_chromosome.genes[i])
        return Chromosome(new_genes)
    
    def two_point_crossover(self,other_chromosome):
        # two point crossover
        crossover_point1 = np.random.randint(0,GENE_LENGTH)
        crossover_point2 = np.random.randint(0,GENE_LENGTH)
        new_genes = []
        for i in range(GENE_LENGTH):
            if i < crossover_point1:
                new_genes.append(self.genes[i])
            elif i < crossover_point2:
                new_genes.append(other_chromosome.genes[i])
            else:
                new_genes.append(self.genes[i])      

        return Chromosome(new_genes)


    def mutate(self,mutation_rate = 0.05):
        
        for gene in self.genes:
            if np.random.randint(0,100) < (mutation_rate * 100):
                gene = np.random.randint(0, 1)

    def decrypt_chromosome(self,genes = None):
        # decrypt the genes and return the weights for the neural network
        global BIT_LENGTH
        weights = []
        for i in range(0,len(self.genes),BIT_LENGTH):
            temp = (np.packbits(self.genes[i:i+BIT_LENGTH]))
            if BIT_LENGTH == 4:
                weights.append(((temp[0]/(2**4)+1)/2**3)-1)
            else:
                weights.append((temp[0]+1)/(2**7)-1)

        return np.array(weights)
    


class InputGenerator():
    def __init__(self,globes=True,stars=True):
        pass
    

    def generate_inputs(self,player_i,pieces,active_player_mask, dice_roll:int):
        
        opponent_players = self.__get_diff_position(player_i,pieces,active_player_mask)
        change_danger = self.__get_change_danger(opponent_players,dice_roll)
        goal_position = self.__get_goal_position(player_i,pieces,active_player_mask)   
        globe_position = self.__get_special_position(player_i,pieces,active_player_mask,"globe")
        star_position = self.__get_special_position(player_i,pieces,active_player_mask,"star")

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

    def __get_diff_position(self,player_i,pieces,mask):
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


    def __get_special_position(self,player_i,pieces,mask,type:str):

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

    def __get_goal_position(self,player_i,pieces,mask):
        # extract pieces from list
        pieces = pieces[0]
        # get pieces of current player
        my_pieces = pieces[player_i]
        # Disregard pieces that are save
        goal_distance = [59 - x for x in my_pieces]
        goal_distance = np.multiply(goal_distance,mask)
        return goal_distance

    def __get_change_danger(self,opponent_players,dice_roll):
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


