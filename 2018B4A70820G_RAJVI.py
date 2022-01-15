import random
import csv
from CNF_Creator import *
import time
import matplotlib.pyplot as plt

#The fitness function calculates the number of clauses in the given 3-CNF sentence satisfied by the model, i.e the individual
def fitness(population,sentence):
    fitvalues=[]
    for individual in population:
        value=0
        for clause in sentence:
            curr = False
            for literal in clause:
                if(literal<0):
                    curr =curr or (not individual[abs(literal)-1])
                else:
                    curr = curr or individual[literal-1]
            if(curr):
                value=value+1
        fitvalues.append(value)
    return fitvalues

#The reproduce functions generate a child model of the new generation by using two parents of the old generation. Here, two methods are used for reproduction

#In the first reproduce function, a crossover point is chosen at random.
#The first values till the crossover point are taken from one parent and the values after the crossover point are taken from the second parent.
#This method generates two children. The child with the greater fitness value is chosen
def reproduce1(parent1, parent2,sentences):
    crossover = random.randrange(0,49)
    child1 = parent1[:crossover]+parent2[crossover:]
    child2 = parent2[:crossover]+parent1[crossover:]
    val = fitness([child1,child2],sentences)
    if(val[0]>val[1]):
        return child1
    return child2

#In the second reproduce function, child indices are randomly chosen to be taken from the first or second parent
#Since the sequence of literals does not impose any structure, this method helps in widening the search space while combining the information from both the parents
#This method also generated two children. The child with the greater fitness value is chosen
def reproduce2(parent1, parent2,sentences):
    crossfn=random.choices([True,False],weights=None,k=50)
    child1=[False]*50
    child2=[False]*50
    for i in range(50):
        if(crossfn[i]==True):
            child1[i]=parent1[i]
            child2[i]=parent2[i]
        else:
            child1[i]=parent2[i]
            child2[i]=parent1[i]
    val = fitness([child1,child2],sentences)
    if(val[0]>val[1]):
        return child1
    return child2

#The mutate function choses an index at random and negates the value present at that index
#This is helps to increase the diversity of the population and hence avoid getting stuck at a local maxima
def mutate(child):
    ind = random.randrange(0,49)
    child[ind]=not child[ind]
    return child

#This function is used to generate the population for the next iteration, given the old and new generations
#Elitism is used by carrying forward the best 10% of the old generation
#Culling is used to stochastically choose the remaining 90% from the entire new generation
def elitism(oldpop,newpop,oldweights,sentences):
    elitezip = zip(oldweights,oldpop)
    elitezip = sorted(elitezip, key = lambda x:x[0])
    tuple1,tuple2 = zip(*elitezip)
    elite = list(tuple2)
    num = len(oldpop)*9//10
    population = elite[num:]
    newweights=fitness(newpop,sentences)
    total = sum(newweights)
    newweights[:]=[x/total for x in newweights]
    culling= random.choices(newpop,newweights,k=num)
    population=population+culling
    return population

#This returns a random population of the given size(default size is 50)
def getRandomPopulation(size=50):
    population=[]
    for i in range(50):
        individual=random.choices([True,False],weights=None,k=size)
        population.append(individual)
    return population

#The genetic algorithm returns the maximum fitness value(expressed as a % of total number of clauses) obtained given the initial population and the 3 CNF sentence 
#The algorithm is terminated when the time exceeds 44 seconds or a model is obtained that satisfies all the clauses
#Random restart is used if the maximum fitness value doesnt change for 100 generations
def genetic_algo(population,sentences):
    start = time.time()
    overallmax=0
    prevmax=0
    count=0
    bestindividual =[]
    for _ in range(10000):
        #break if time exceeds 44 seconds
        if(time.time()-start>44):
            break
        weights = fitness(population,sentences)
        #the maximum value obtained by the algorithm is tracked
        currmax=max(weights)/len(sentences)*100
        if(currmax>overallmax):
            overallmax=currmax
            index=weights.index(max(weights))
            bestindividual=population[index]
        prevmax=currmax
        #the weights express the fitness value as a fraction of sum of fitness values of all the individuals in the population
        total = sum(weights)
        weights[:]=[x/total for x in weights]        
        if currmax==100:
            break
        if currmax==prevmax:
            count=count+1
        else:
            count=1
        #If the maximum value obtained by a generation does not change for 100 consecutive generations, random restart is applied
        if count>100:
            count=0
            population=getRandomPopulation(50)
            weights = fitness(population,sentences)
            total = sum(weights)
            weights[:]=[x/total for x in weights] 
        pop2=[]
        #Reproduction is carried out by stochastically choosing one of the reproduce functions, with a 10%probability of using the random sequence and 90% probability of choosing a single crossover point
        for i in range(len(population)):
            parent1,parent2=random.choices(population,weights,k=2)
            if(random.random()>0.5):
                child = reproduce1(parent1,parent2,sentences)
            else:
                child = reproduce2(parent1,parent2,sentences)
            #The child is mutated with 10% probability    
            if(random.random()<0.1):
                child = mutate(child)
            pop2.append(child)
        #elitism and culling are used to choose the population for the next iteration 
        population=elitism(population,pop2,weights,sentences)
    weights = fitness(population,sentences)
    currmax = max(weights)/len(sentences)*100
    if(currmax>overallmax):
        overallmax=currmax
        index=weights.index(max(weights))
        bestindividual=population[index]
    return overallmax,bestindividual

def randomSentence(clauses):
    cnfC = CNF_Creator(n=50) # n is number of symbols in the 3-CNF sentence
    sentence = cnfC.CreateRandomSentence(m=clauses) # m is number of clauses in the 3-CNF sentence
    #print('Random sentence : ',sentence)
    start_time=time.time()
    population=getRandomPopulation(50)
    accuracy,bestindividual=genetic_algo(population,sentence)
    for i in range(50):
        if(bestindividual[i]==True):
            bestindividual[i]=i+1
        else:
            bestindividual[i]=-(i+1)
    end_time=time.time()
    timetaken=end_time-start_time
    print('Number of clauses in CSV file : ',len(sentence))
    print('Best model : ',bestindividual)
    print('Fitness value of best model : ',accuracy,'%')
    print('Time taken :', timetaken, 'seconds')

def CSVFileCNF():
    cnfC = CNF_Creator(n=50) # n is number of symbols in the 3-CNF sentence
    sentence = cnfC.ReadCNFfromCSVfile()
    #print('\nSentence from CSV file : ',sentence)
    start_time=time.time()
    population=getRandomPopulation(50)
    accuracy,bestindividual=genetic_algo(population,sentence)
    for i in range(50):
        if(bestindividual[i]==True):
            bestindividual[i]=i+1
        else:
            bestindividual[i]=-(i+1)
    end_time=time.time()
    timetaken=end_time-start_time
    print('Number of clauses in CSV file : ',len(sentence))
    print('Best model : ',bestindividual)
    print('Fitness value of best model : ',accuracy,'%')
    print('Time taken :', timetaken, 'seconds')    
def main():
    print('\n\n')
    print('Roll No : 2018B4A70820G')
#    randomSentence(120)
    CSVFileCNF()
    print('\n\n')
    
if __name__=='__main__':
    main()