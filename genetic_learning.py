import numpy as np

GENE_LENGTH = 32

class GeneticPlayer:
    def __init__(self,gene = []):
        self.parent = False
        if not gene:
            for i in range(GENE_LENGTH):
                self.gene.append(np.random.randint(0, 1))
        else:
            self.gene = gene

    def decode_action():
        print("What am i doing here?")

