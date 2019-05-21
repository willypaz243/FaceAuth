import numpy as np
class K_mean():
    """Modelo de agrupacion"""
    def __init__(self):
        self.grupos = 0
        self.centroides = None
        self.nombres = []
    
    def train(self, dataset):
        """
        Mueve los puntos centrales "centroides" a una distancia media 
        de los datos de cada dato del dataset.
        """
        centroides_ant = np.copy(self.centroides)
        print("TRAINING...")
        while centroides_ant - self.centroides > 0:
            centroides_ant = np.copy(self.centroides)
            distancias = []
            conjuntos = []
        for i in range(self.grupos):
            distancias.append(((dataset-self.centroides[i])**2).sum(axis=1)**0.5)
        distancias = np.array(distancias)
        booleanos = distancias.min(axis=0) == distancias
        for i, value in enumerate(booleanos):
            self.centroides[i] = np.mean(dataset[value], axis=0)
        print("FINISHED TRAINING")
    def set_centroides(self, centroides):
        self.centroides = centroides
        self.grupos = self.centroides.shape[0]

    def set_nombre(self, nombres):
        self.nombres = nombres

    def that_class(self, entrada):
        """
        devuelve el nombre del grupo al que pertenece un punto de entrada.
        """
        distancias = list(((self.centroides-entrada)**2).sum(axis=1))
        minimo = min(distancias)
        indice = distancias.index(minimo)
        if minimo < 0.6:
            return self.nombres[indice]
        else:
            return "Desconocido"
