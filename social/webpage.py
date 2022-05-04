import numpy as np
from enum import Enum

from sympy import ground_roots

class ProductId(Enum):
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4
    P5 = 5

class User:
    def __init__(self, alpha:dict, reservationPrices:dict):
        self.alpha = alpha
        self.reservationPrices = reservationPrices

class Product:
    def __init__(self, id:ProductId, price):
        self.id = id
        self.price = price

class WebpageSession:
    def __init__(self, products, matrix:np.matrix, user:User):
        self.products = products;
        self.__visitedProducts = []
        self.__incidenceMatrix = matrix
        self.__user = user

    def buyProduct(self, p:Product):
        self.__visitedProducts.append(p)
        #if(p.price < self.__user.reservationPrices[p.id]):
            #pesco il primo secondary
            #row = self.
            #con prob p1 clicco
                #se clicco, visualizzo pagina di p1
                #buyProduct(p1)
                #altro
                #return
            #pesco secondo secondary
            #se non clicco, con prob p2*lambda clicco pagina di p2
                #se clicco, visualizzo pagina di p2
                #buyProduct(p2)
                #altro
                #return
            #qui non ho cliccato su nessun secondary
            #return


if __name__ == "__main__":

    prob_matrix = np.random.uniform(0.0, 0.5, (5, 5))

    n_days = 100
    max_users = 1000
    for d in range(n_days):
        n_user = np.random.randint(1, max_users)
        alphas = np.random.dirichlet([1, 1, 1, 1, 1, 1])
        user_choices = np.floor(alphas * n_user)
        for c in range(1, len(user_choices)):
            #skip alpha 0
            group = user_choices[c]
            for i in range(0, group):
                #SIMULATE PURCHASE OF PRODUCTS
            

        
