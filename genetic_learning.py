import numpy as np
import input_generator
import ludopy

NUMBER_OF_HIDDEN_NEURONS = 5
NUMBER_OF_INPUTS = 9
NUMBER_OF_WEIGHTS = NUMBER_OF_INPUTS * NUMBER_OF_HIDDEN_NEURONS + NUMBER_OF_HIDDEN_NEURONS
BIT_LENGTH = 8
GENE_LENGTH = BIT_LENGTH * NUMBER_OF_WEIGHTS



class GeneticPlayer:

    global GENE_LENGTH

    def __init__(self,population_size = 100, fitness_loops = 20, mutation_rate = 0.05, crossover_rate = 0.5, selection_rate = 0.5, selection_type = "tournament"):
        self.variants = []
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_rate = selection_rate
        self.selection_type = selection_type
        self.generation = 0
        self.fitness_loops = fitness_loops
        self.game = ludopy.Game()
        self.__generate_initial_population()
        self.__calculate_fitness()
        self.__sort_population()
    
    def __generate_initial_population(self):
        for i in range(self.population_size):
            self.variants.append(Chromosome(np.random.randint(0, 1, size=GENE_LENGTH)))

    def __calculate_fitness(self):
        for variant in self.variants:
            pass

    def __sort_population(self):
        self.variants.sort(key=lambda x: x.fitness, reverse=True)

    def __compute_fitness(self,players="random"):
        # fitness for all chromosomes in the current generation
        for chromosome in self.variants:
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
                            I = input_generator.generate_inputs(player_i,pieces,mask,dice)

                            activation = self.__run_neural_networks(I,weights)

                            #piece_to_move = function call
                            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))] # to be replaced by function call

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


                    _, _, _, _, _, there_is_a_winner = self.game.answer_observation(piece_to_move)

                if player_i == 0:
                    winning_count += 1
                print("Player: ", player_i," won")
            
        
            # fitness_loop amount of games have been played
            chromosome.fitness = winning_count/self.fitness_loops



    def __run_neural_networks(input_matrix,weights,activation_function = "ReLU"):
        global NUMBER_OF_HIDDEN_NEURONS
        pass
        
    def __selcect_parents(self):
        if self.selection_type == "tournament":
            return self.__tournament_selection()
        else:
            return self.__roulette_wheel_selection()    

    def train(self,generations = 100):
        for i in range(generations):
            self.__compute_fitness()
            self.generation += 1



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

    def decrypt_chromosome(self):
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