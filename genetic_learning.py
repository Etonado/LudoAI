import numpy as np
import input_generator
import ludopy
from multiprocessing import Pool
import time



NUMBER_OF_HIDDEN_NEURONS = 10
NUMBER_OF_INPUTS = 9
NUMBER_OF_WEIGHTS = NUMBER_OF_INPUTS * NUMBER_OF_HIDDEN_NEURONS + NUMBER_OF_HIDDEN_NEURONS
BIT_LENGTH = 8
GENE_LENGTH = BIT_LENGTH * NUMBER_OF_WEIGHTS
MULTI_CORE = True 
NUMBER_OF_CORES = 4 # can't be changed atm :/



class GeneticPlayer:

    global GENE_LENGTH, MULTI_CORE, NUMBER_OF_CORES

    def __init__(self,population_size = 100, fitness_loops = 400, mutation_rate = 0.1, crossover_rate = 0.5, selection_rate = 0.5, selection_type = "tournament"):
        self.variants = []
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_rate = selection_rate
        self.generation = 0
        self.elite_size = int(0.1*population_size)
        self.parent_size  = int(0.2*population_size)
        self.selection_type = selection_type
        self.tournament_size = 4
        self.fitness_loops = fitness_loops
        self.game = ludopy.Game()
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
                            I = input_generator.generate_inputs(player_i,pieces,mask,dice)
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

        if MULTI_CORE:
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
        if MULTI_CORE:
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

    def train(self,generations = 100):
        # compute fitness of initial population
        time_start = time.time()
        self.__call_fitness_function()
        time_end = time.time()
        print("Time to compute fitness of initial population: ", time_end-time_start)
        self.generation = 1
        while self.generation < generations:
            # sort population
            self.__sort_population()
            # print average fitness of previous generation
            print("Average fitness of generation ", self.generation-1,": ", self.__average_fitness())
            # print max fitness of generation
            print("Generation: ", self.generation -1, " Max fitness: ", self.variants[0].fitness)
            # print lowest fitness of generation
            print("Generation: ", self.generation -1, " Min fitness: ", self.variants[self.population_size-1].fitness)

            # save statistics
            with open('./statistics/file1.txt', 'a') as file:
                file.write("%d,%.2f,%.2f,%.2f\n" % (self.generation-1,self.variants[self.population_size-1].fitness,self.variants[0].fitness,self.__average_fitness()))

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
        with open('./weights/best_5_current_gen.txt', 'r') as file:
            genes = np.loadtxt(file, delimiter=',', max_rows=1).astype(int)
        
                
        winner = Chromosome(genes)
       
        weights = winner.decrypt_chromosome()

        return weights



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
    
