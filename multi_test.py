from multiprocessing import Pool
import numpy as np
import time

def __my_function(x,y,a):
        # function implementation
        b = x+y
        print(b)
        print(a)
        return b


    

if __name__ == '__main__':
    start_time = time.time()
    with Pool(processes=2) as pool:
        inputs = [(10,20,"f"), (10,30,"k")]
        results = pool.starmap(__my_function, inputs)
        print(results)
    
    print(time.time()-start_time)
 