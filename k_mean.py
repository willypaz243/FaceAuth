import numpy as np
class K_mean():
    """Modelo de agrupacion"""
    def __init__(self):
        self.grupos = 0
        self.centroides = np.array([])
        self.nombres = []
        self.radios = []
    
    def train(self, dataset):
        """
        Mueve los puntos centrales "centroides" a una distancia media 
        de los datos de cada dato del dataset.
        """
        dimenciones = self.centroides.shape
        centroides_ant = np.random.rand(dimenciones[0],dimenciones[1])*6
        print("TRAINING...")
        diferencia = centroides_ant - self.centroides
        while diferencia.max() > 0:
            diferencia = centroides_ant - self.centroides
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
        self.load_radios()

    def set_nombre(self, nombres):
        self.nombres = nombres

    def that_class(self, entrada):
        """
        devuelve el nombre del grupo al que pertenece un punto de entrada.
        """
        if not self.centroides.size == 0:
            distancias = list(((self.centroides-entrada)**2).sum(axis=1))
            minimo = min(distancias)
            indice = distancias.index(minimo)
            if minimo < self.radios[indice]:
                return self.nombres[indice]
            else:
                return "Desconocido"
        else:
            return "Desconocido"

    def load_radios(self):
        print(self.centroides)
        if self.grupos > 1:
            radios = []
            for i, v in enumerate(self.centroides):
                centros = self.centroides.tolist()
                centros.remove(list(v))
                centros = np.array(centros)
                radio = ((v - centros)**2).sum(axis=1).min()/2
                radios.append(radio)
            self.radios = radios
        else:
            self.radios.append(0.2)