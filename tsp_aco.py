# ------------------------------------------------------------------------------+
#
#   Matteo Tortora
#   Solution for Travelling Salesman Problem using ACO (Ant Colony Optimization)
#   July, 2019
#
# ------------------------------------------------------------------------------+

from math import sin, cos, sqrt, atan2, radians
import numpy as np
import pandas as pd

# class that represents a World


class World:
    def __init__(self, cities=None):
        if cities is None:
            self.cities = {}  # dictionary of cities
            self.edges = {}   # dictionary of edges
        else:
            self.cities = cities
            self.numcities = len(self.cities.keys())
            self.createCompleteGraph()
            self.createDistancematrix()

    # adds a new city with coordinates and a cost
    def addCity(self, namecity, lat, lon):
        # checks if the edge already exists
        if not self.existsCity(namecity):
            self.cities[namecity] = [lat, lon]
            self.numcities = len(self.cities.keys())
        else:
            print(f"{namecity} already exists!")

    def deleteCity(self, namecity):
        try:
            del(self.cities[namecity])
            print(f"{namecity} successfully deleted")
        except KeyError:
            print(f"{namecity} not found")

    def createDistanceMatrix(self):
        self.distmatrix = pd.DataFrame(columns=list(
            self.cities.keys()), index=list(self.cities.keys()))
        listcities = list(self.cities.keys())
        for i in range(self.numcities):
            for j in range(self.numcities):
                self.distmatrix.iloc[i, j] = self.distcities(listcities[i], listcities[j])
                self.distmatrix.iloc[j, i] = self.distmatrix.iloc[i, j]

    def createCompleteGraph(self):
        weight = 1/self.numcities
        listcities = list(self.cities.keys())
        for i in range(self.numcities):
            edge = {}
            for j in range(self.numcities):
                if j != i:
                    edge[listcities[j]] = weight
            self.edges[listcities[i]] = edge

    def distcities(self, src, dest):
        R = 6373.0  # approximate radius of earth in km

        lat1 = radians(self.cities[src][0])
        lon1 = radians(self.cities[src][1])
        lat2 = radians(self.cities[dest][0])
        lon2 = radians(self.cities[dest][1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        distance = R*(2 * atan2(sqrt(a), sqrt(1 - a)))

        return distance

    # checks if exists a City
    def existsCity(self, namecity):
        return (True if namecity in self.cities.keys() else False)

    # shows all the city in the world
    def showWorld(self):
        print('Showing the cities in the World:\n')
        for city in self.cities.keys():
            print(
                f"{city} with latitude: {self.cities[city][0]} and longitude: {self.cities[city][0]}")


class AntColony:
    def __init__(self, n_ants, n_iterations, decay=0.6, alpha=1, beta=1):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.bpath = []

    def createPherorMatrix(self, world):
        self.pheromone = pd.DataFrame(columns=list(world.cities.keys()),
                                      index=list(world.cities.keys()))
        for i in range(len(world.cities.keys())):
            for j in range(len(world.cities.keys())):
                self.pheromone.iloc[i, j] = 1/len(world.cities.keys())
                self.pheromone.iloc[j, i] = self.pheromone.iloc[i, j]

    def createColony(self):
        self.colony = {}
        for i in range(self.n_ants):
            self.colony[i] = []

    def initializeColony(self, world):
        if self.n_ants == len(world.cities.keys()):
            for i, city in zip(range(self.n_ants), world.cities.keys()):
                self.colony[i] = [city]
        else:
            for i in range(self.n_ants):
                self.colony[i] = list(world.cities.keys())[np.random.randint(world.numcities)]

    def probaij(self, tauij, etaij,):
        return proba

    def run(self, world):
        self.createColony()
        self.createPherorMatrix(word)
        for i in range(self.n_iterations):
            self.initializeColony(world)
            unvisitedcity = list(world.cities.keys())
            for ant in range(self.n_ants):
                unvisitedcity = unvisitedcity.remove(self.colony[ant][0])
                for j in range(world.numcities):
                    # trova path ant
                    proba = []
                for city, w in unvisited:
                    proba[city]
                    # tutti proba

            self.bpath
            # Update Pherormone
