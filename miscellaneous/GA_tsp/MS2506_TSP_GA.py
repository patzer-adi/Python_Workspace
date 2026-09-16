"""
MS2506:Travelling Salesman Problem using Genetic Algorithm

Input file:
    india_cities_GA.txt

The input file should contain one Indian city name per line.

Method:
    1. Read city names
    2. Obtain latitude and longitude using geopy
    3. Construct Euclidean distance matrix
    4. Generate an initial population of routes
    5. Calculate route distance and fitness
    6. Select parents using roulette-wheel selection
    7. Apply Order Crossover (OX)
    8. Apply swap mutation
    9. Repeat for the specified number of generations
   10. Plot the final route and fitness history
"""

import random
import time

import numpy as np
import matplotlib.pyplot as plt

from geopy.geocoders import Nominatim


class GeneticAlgorithmTSP:
    """
    Genetic Algorithm solution for the Travelling Salesman Problem.
    """

    def __init__(
        self,
        filename="india_cities_GA.txt",
        population_size=100,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.02,
        random_seed=42
    ):

        self.filename = filename

        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_seed = random_seed

        self.cities = []
        self.coordinates = {}
        self.distance_matrix = None

        self.population = []

        self.best_route = None
        self.best_distance = None

        self.average_fitness_history = []


    def load_cities(self):
        """
        Read city names from the input file.
        """

        with open(self.filename, "r", encoding="utf-8") as file:

            for line in file:

                city = line.strip()

                if city:
                    self.cities.append(city)

        if len(self.cities) < 2:
            raise ValueError("At least two cities are required.")


    def get_coordinates(self):
        """
        Obtain latitude and longitude for every city.
        """

        geolocator = Nominatim(
            user_agent="genetic_algorithm_tsp"
        )

        print("\nGetting city coordinates...\n")

        for number, city in enumerate(self.cities, start=1):

            print(
                f"[{number}/{len(self.cities)}] "
                f"Getting coordinates for {city}...",
                end=" "
            )

            location = geolocator.geocode(
                f"{city}, India"
            )

            if location is None:
                print("FAILED")
                raise ValueError(
                    f"Could not find coordinates for {city}"
                )

            latitude = round(location.latitude, 2)
            longitude = round(location.longitude, 2)

            self.coordinates[city] = (
                latitude,
                longitude
            )

            print(
                f"OK -> ({latitude}, {longitude})"
            )

            time.sleep(1)

        print("\nAll cities found.\n")


    def calculate_distance(self, coordinate_1, coordinate_2):
        """
        Calculate Euclidean distance between two coordinates.
        """

        latitude_1, longitude_1 = coordinate_1
        latitude_2, longitude_2 = coordinate_2

        distance = np.sqrt(
            (latitude_1 - latitude_2) ** 2
            +
            (longitude_1 - longitude_2) ** 2
        )

        return distance


    def create_distance_matrix(self):
        """
        Create the distance matrix.
        """

        number_of_cities = len(self.cities)

        self.distance_matrix = np.zeros(
            (number_of_cities, number_of_cities)
        )

        for i in range(number_of_cities):

            for j in range(i + 1, number_of_cities):

                city_1 = self.cities[i]
                city_2 = self.cities[j]

                distance = self.calculate_distance(
                    self.coordinates[city_1],
                    self.coordinates[city_2]
                )

                self.distance_matrix[i][j] = distance
                self.distance_matrix[j][i] = distance


    def create_population(self):
        """
        Create the initial population.

        A route is represented by a list of city indices.
        """

        number_of_cities = len(self.cities)

        self.population = []

        for i in range(self.population_size):

            route = list(range(number_of_cities))

            random.shuffle(route)

            self.population.append(route)


    def calculate_route_distance(self, route):
        """
        Calculate the total distance of a route.
        """

        total_distance = 0.0

        for i in range(len(route) - 1):

            city_1 = route[i]
            city_2 = route[i + 1]

            total_distance += self.distance_matrix[
                city_1
            ][
                city_2
            ]

        # The salesman has to come back home.
        total_distance += self.distance_matrix[
            route[-1]
        ][
            route[0]
        ]

        return total_distance


    def calculate_fitness(self, distance):
        """
        Calculate fitness from route distance.

        Shorter route = better fitness.
        """

        return 1.0 / distance


    def evaluate_population(self):
        """
        Calculate distance and fitness of every route.
        """

        distances = []
        fitnesses = []

        for route in self.population:

            distance = self.calculate_route_distance(route)
            fitness = self.calculate_fitness(distance)

            distances.append(distance)
            fitnesses.append(fitness)

        return distances, fitnesses


    def select_parents(self, fitnesses):
        """
        Select two parents using roulette-wheel selection.
        """

        total_fitness = sum(fitnesses)

        probabilities = []

        for fitness in fitnesses:

            probability = fitness / total_fitness

            probabilities.append(probability)

        parent_indices = np.random.choice(
            len(self.population),
            size=2,
            replace=False,
            p=probabilities
        )

        parent_1 = self.population[
            parent_indices[0]
        ].copy()

        parent_2 = self.population[
            parent_indices[1]
        ].copy()

        return parent_1, parent_2


    def crossover(self, parent_1, parent_2):
        """
        Perform Order Crossover (OX).
        """

        if random.random() > self.crossover_rate:

            return parent_1.copy(), parent_2.copy()

        length = len(parent_1)

        start, end = sorted(
            random.sample(
                range(length),
                2
            )
        )

        child_1 = [-1] * length
        child_2 = [-1] * length

        # Copy a part of each parent.
        child_1[start:end] = parent_1[start:end]
        child_2[start:end] = parent_2[start:end]

        # Fill child 1 using parent 2.
        position = end

        for city in parent_2:

            if city not in child_1:

                if position == length:
                    position = 0

                child_1[position] = city
                position += 1

        # Fill child 2 using parent 1.
        position = end

        for city in parent_1:

            if city not in child_2:

                if position == length:
                    position = 0

                child_2[position] = city
                position += 1

        return child_1, child_2


    def mutate(self, route):
        """
        Apply swap mutation.
        """

        new_route = route.copy()

        if random.random() < self.mutation_rate:

            index_1, index_2 = random.sample(
                range(len(new_route)),
                2
            )

            new_route[index_1], new_route[index_2] = (
                new_route[index_2],
                new_route[index_1]
            )

        return new_route


    def create_next_generation(self, fitnesses):
        """
        Create the next generation.

        Selection -> Crossover -> Mutation
        """

        new_population = []

        while len(new_population) < self.population_size:

            parent_1, parent_2 = self.select_parents(
                fitnesses
            )

            child_1, child_2 = self.crossover(
                parent_1,
                parent_2
            )

            child_1 = self.mutate(child_1)
            child_2 = self.mutate(child_2)

            new_population.append(child_1)

            if len(new_population) < self.population_size:

                new_population.append(child_2)

        self.population = new_population


    def run(self):
        """
        Run the complete Genetic Algorithm.
        """

        if self.random_seed is not None:

            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        print("\nLoading cities...")
        self.load_cities()

        self.get_coordinates()

        print("Creating distance matrix...")
        self.create_distance_matrix()

        print("Creating initial population...")
        self.create_population()

        print("\nStarting Genetic Algorithm...\n")

        for generation in range(self.generations):

            distances, fitnesses = (
                self.evaluate_population()
            )

            average_fitness = np.mean(fitnesses)

            self.average_fitness_history.append(
                average_fitness
            )

            best_index = np.argmax(fitnesses)

            if self.best_distance is None:

                self.best_distance = distances[best_index]

                self.best_route = (
                    self.population[best_index].copy()
                )

            elif distances[best_index] < self.best_distance:

                self.best_distance = distances[best_index]

                self.best_route = (
                    self.population[best_index].copy()
                )

            self.create_next_generation(
                fitnesses
            )

            if (
                generation == 0
                or (generation + 1) % 50 == 0
                or generation == self.generations - 1
            ):

                print(
                    f"Generation {generation + 1}: "
                    f"Best Distance = "
                    f"{self.best_distance:.4f}"
                )

        print("\nGenetic Algorithm finished.\n")


    def display_result(self):
        """
        Display the final result.
        """

        if self.best_route is None:

            print("Run the genetic algorithm first.")

            return

        route_names = []

        for city_index in self.best_route:

            route_names.append(
                self.cities[city_index]
            )

        # Return to the starting city.
        route_names.append(route_names[0])

        print("\n" + "=" * 60)
        print("GENETIC ALGORITHM - TRAVELLING SALESMAN PROBLEM")
        print("=" * 60)

        print(
            f"\nNumber of cities : {len(self.cities)}"
        )

        print(
            f"Population size  : {self.population_size}"
        )

        print(
            f"Generations      : {self.generations}"
        )

        print(
            f"Crossover rate   : {self.crossover_rate}"
        )

        print(
            f"Mutation rate    : {self.mutation_rate}"
        )

        print("\nBest route:")

        print(" -> ".join(route_names))

        print(
            f"\nMinimum distance : "
            f"{self.best_distance:.4f}"
        )

        print("=" * 60)


    def plot_route(self):
        """
        Plot the best route.
        """

        if self.best_route is None:

            print("No route available.")

            return

        route = self.best_route.copy()

        # Close the route.
        route.append(route[0])

        latitudes = []
        longitudes = []

        for city_index in route:

            city = self.cities[city_index]

            latitude, longitude = (
                self.coordinates[city]
            )

            latitudes.append(latitude)
            longitudes.append(longitude)

        plt.figure(figsize=(12, 8))

        plt.plot(
            longitudes,
            latitudes,
            marker="o"
        )

        for order, city_index in enumerate(
            self.best_route
        ):

            city = self.cities[city_index]

            latitude, longitude = (
                self.coordinates[city]
            )

            plt.annotate(
                f"{order + 1}. {city}",
                (longitude, latitude),
                xytext=(5, 5),
                textcoords="offset points"
            )

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.title(
            "Best Route using Genetic Algorithm"
        )

        plt.grid(True)
        plt.tight_layout()

        plt.show()


    def plot_fitness_history(self):
        """
        Plot average fitness against generation.
        """

        if not self.average_fitness_history:

            print("No fitness history available.")

            return

        generations = range(
            1,
            len(self.average_fitness_history) + 1
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            generations,
            self.average_fitness_history
        )

        plt.xlabel("Generation")
        plt.ylabel("Average Fitness")

        plt.title(
            "Average Fitness vs Generation"
        )

        plt.grid(True)
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":

    ga = GeneticAlgorithmTSP(
        filename="india_cities_GA.txt",
        population_size=100,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.02,
        random_seed=42
    )

    ga.run()

    ga.display_result()

    ga.plot_route()

    ga.plot_fitness_history()