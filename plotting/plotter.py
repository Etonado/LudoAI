import csv
import matplotlib.pyplot as plt
import numpy as np


number = 4

if number == 0:
    # Read the CSV file line by line
    with open('./10n_rate.csv', 'r') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            data = row#(float(row[0]))

    data = np.array(data).astype(float)
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    # print statistics
    print("Mean: ", mean)
    print("Std: ", std)
    plt.figure()
    plt.hist(data)
    plt.show()

elif number == 1:
    # Read the CSV file line by line
    with open('./6n_rate.csv', 'r') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            data = row#(float(row[0]))

    data = np.array(data).astype(float)
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    # print statistics
    print("Mean: ", mean)
    print("Std: ", std)
    plt.figure()
    plt.hist(data)
    plt.show()


elif number == 2:
    mean = []
    std = []
    max = []

    # Read the CSV file line by line
    with open('./statistics/file2.txt', 'r') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            max.append(float(row[2]))
            mean.append(float(row[3]))
            std.append(float(row[4]))

    # Plotting
    plt.figure()
    # print legend
    
    plt.plot(max,'o--',label='Best performing player')
    plt.plot(mean,'b',label='Average player of the population')
    # plot graph with mean + std
    plt.plot([i+j for i,j in zip(mean,std)],'r--',label='Standard Deviation of the population')
    plt.plot([i-j for i,j in zip(mean,std)],'r--')
    plt.plot()
    plt.legend()
    plt.ylabel('Winning rate')
    plt.xlabel('Generation')
    plt.title('Winning rate per generation')
    plt.grid()
    axis = plt.gca()
    axis.set_ylim([0,0.6])

    plt.show()

elif number == 3:
    data = []
    # Read the CSV file line by line
    with open('./winning_rate.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
             sav = row
    for entries in sav:
        data.append(float(entries))

    # Plotting
    plt.figure()
    # print legend
    
    plt.plot(data)
    plt.ylabel('Winning rate')
    plt.xlabel('Games played')
    plt.title('Winning rate depending on the number of games played')
    plt.grid()
    axis = plt.gca()
    axis.set_ylim([0,1])
    axis.set_xlim([-10,2000])
    plt.show()

elif number == 4:
    # Read the CSV file line by line
    with open('./d_rate.csv', 'r') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            d = row#(float(row[0]))

    with open('./e_rate.csv', 'r') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            e = row
    
    with open('./r_rate.csv', 'r') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            r = row

    d = np.array(d).astype(float)
    e = np.array(e).astype(float)
    r = np.array(r).astype(float)

    # divide all elements of the array by 300
    d = d/300   
    e = e/300
    r = r/300

    d_mean = np.mean(d,axis=0)
    d_std = np.std(d,axis=0)
    e_mean = np.mean(e,axis=0)
    e_std = np.std(e,axis=0)
    r_mean = np.mean(r,axis=0)
    r_std = np.std(r,axis=0)

    # print statistics
    print("Mean: ", d_mean)
    print("Std: ", d_std)
    print("Mean: ", e_mean)
    print("Std: ", e_std)
    print("Mean: ", r_mean)
    print("Std: ", r_std)
    plt.figure()
    plt.grid()
    plt.ylabel('Number of observed winning rates')
    plt.xlabel('Winning rate')
    plt.hist(r, label='Random')
    plt.hist(e, label='Genetic algorithm')
    plt.hist(d, label='Q-learning')
    plt.legend()
    plt.title('Distribution of winning rates')
    plt.show()
