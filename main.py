#wzmacnieniae obrazu metodą rozmytej entropii za pomocą algorytmu genetycznego
import cv2
import numpy as np
import random
import math

MAX_PIXEL_VALUE = 255
POPULATION_SIZE = 100
FINAL_GREY_LEVELS = 3
GENERATIONS = 300

def histogram_img(img):
    hist = np.zeros(MAX_PIXEL_VALUE + 1, dtype=int)
    for row in img:
        for pixel in row:
            hist[pixel]+=1
    return hist

def generate_population(high, size):
    # tworzenie popolacji jako tablicy
    population = np.array([np.random.randint(high, size=3) for _ in range(size)], np.uint8)
    # sortowanie hromosonów populacji od najmniejszego do najwiekszego
    population = np.sort(population, axis=1)
    return population

def crossover(element1, element2, probability):
    # sprawdzenie czy wystąpi krzyżowanie
    if random.uniform(0, 1) < probability:
        # utworzenie tablicy
        new_element1 = np.array([], np.uint8)
        new_element2 = np.array([], np.uint8)
        for d in range(3):
            # losowanie punktu krzyżowania
            crossover_point = random.randint(1, 7)
            # przetwarzanie na postać bitową
            bits1 = np.unpackbits(element1[d])
            bits2 = np.unpackbits(element2[d])
            # wykonanie krzyżowania
            new_bits1 = np.concatenate([bits1[0:crossover_point], bits2[crossover_point:8]], axis=0)
            new_bits2 = np.concatenate([bits1[0:crossover_point], bits2[crossover_point:8]], axis=0)
            # zamiana na int
            new_element1 = np.append(new_element1, np.packbits(new_bits1))
            new_element2 = np.append(new_element2, np.packbits(new_bits2))
        element2 = new_element2
        element1 = new_element1
    return element1, element2

def mutation(element, probability):
    if random.uniform(0, 1) < probability:
        # odczytanie ilosci wymiarow
        dimensions = len(element)
        # losowanie wymiaru
        random_dimension = random.randint(0, dimensions - 1)
        # losowanie bitu
        random_bit = random.randint(1, 7)
        # przetwarzanie na postać bitową
        bits = np.unpackbits(element)
        # wykonanie mutacji
        bits[random_dimension * 8 + random_bit] = (bits[random_dimension * 8 + random_bit] + 1) % 2
        element = np.packbits(bits)
    return element

def adjust_values(element):
    #poprawa wartości hromosonów
    if (element[0] > element[1]):
        element[0] = (element[1] - 1) * (element[0] / 255)
    if (element[1] > element[2]):
        element[2] = (element[1] + 1) + (254 - element[1]) * (element[2] / 255)
    return element

def membership_function(element):
    #tworzenie osobnika
    membership_array = np.array([])
    element = element.astype(np.float32)
    for i in range(MAX_PIXEL_VALUE+1):
            if (i <= element[0]):
                membership_array = np.append(membership_array, 0)
            elif (element[0] <= i and i <= element[1]):
                membership_array = np.append(membership_array,((i - element[0]) ** 2) / (
                            (element[1] - element[0]) * (element[2] - element[0])))
            elif (element[1] <= i and i <= element[2]):
                membership_array = np.append(membership_array, 1 - ((i - element[2]) ** 2) / (
                        (element[2] - element[1]) * (element[2] - element[0])))
            else:
                membership_array = np.append(membership_array, 1)
    return membership_array

def calculate_entropy(Pp, FINAL_GREY_LEVELS):
    x = 0
    for i in range(FINAL_GREY_LEVELS):
        if(Pp[i] != 0):
            x = x + Pp[i] * math.log10(Pp[i])
    H = -x / (math.log10(3))
    return H

def calculate_Pp(FINAL_GREY_LEVELS, prob_hist, membership):
   Pp = np.array([])
   for i in range(FINAL_GREY_LEVELS):
       condition_checked = (np.where((membership <= (i+1)/FINAL_GREY_LEVELS) & (membership > i / FINAL_GREY_LEVELS), 1, 0))
       Pp = np.append(Pp, np.sum(prob_hist * condition_checked))

   return Pp
def imageprocesing(img):
    x, y = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] <= gray_bounds[0]:
                img[i][j] = 0
            elif img[i][j] >= gray_bounds[1]:
                img[i][j] = 255
            else:
                img[i][j] = int(255 / 2)
    return img
if __name__ == "__main__":


    filenames = ("example3.jpg")  # , "example2.jpg", "example3.jpg")
    img = cv2.imread(filenames, cv2.IMREAD_GRAYSCALE)
    hist_img = histogram_img(img)
    parents_table = np.array([], np.uint8)

    old_pop = generate_population(MAX_PIXEL_VALUE + 1, POPULATION_SIZE)
    probability_histogram = hist_img/np.sum(hist_img).astype(np.float32)
    # start petli
    for k in range(GENERATIONS):
        new_pop = []
        for j in range(100):
            membership_array = membership_function(old_pop[j])
            Pp = calculate_Pp(FINAL_GREY_LEVELS, probability_histogram, membership_array)
            entropy = round(calculate_entropy(Pp, FINAL_GREY_LEVELS), 2)
            parent_probaility = int(entropy * 10)
            for p in range(parent_probaility):
                parents_table = np.append(parents_table, j)
        for j in range(50):
            parent_1 = random.choice(parents_table)
            parent_2 = random.choice(parents_table)
            while(parent_2==parent_1):
                parent_2 = random.choice(parents_table)
            child1, child2 = crossover(old_pop[parent_1], old_pop[parent_2], 0.5)
            child1 = mutation(child1, 0.01)
            child2 = mutation(child2, 0.01)
            new_pop.append(adjust_values(child1))
            new_pop.append(adjust_values(child2))
        old_pop = np.array(new_pop, np.uint8)
    entropy_list = []
    membership_list = []


    for j in range(100):

        membership = membership_function(old_pop[j])
        membership_list.append(membership)
        Pp = calculate_Pp(FINAL_GREY_LEVELS, probability_histogram, membership_array)
        entropy_list.append(round(calculate_entropy(Pp, FINAL_GREY_LEVELS), 2))
    best_parameters = old_pop[np.argmax(entropy_list)]
    best_membership = membership_list[np.argmax(entropy_list)]
    print(np.max(entropy_list))

    gray_bounds = []
    for i in range(FINAL_GREY_LEVELS-1):
         membership_condition = np.where(best_membership <= (i+1)/FINAL_GREY_LEVELS, best_membership, 0)
         gray_bounds.append(np.argmax(membership_condition))
    print(gray_bounds)

    img = imageprocesing(img)
    cv2.imshow("processed image", img)
    cv2.waitKey(0)