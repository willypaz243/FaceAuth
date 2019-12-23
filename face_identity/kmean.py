import os
import numpy as np
from pymongo import MongoClient

class K_mean():
    """Modelo de agrupacion de clases"""
    def __init__(self, model_name):
        self.model_name = model_name
        
        self.grupos = 0
        self.centroides = np.array([])
        self.ids = np.array([])
        self.radios = 0.01
        self.dataset = np.array([])
        self.load_model()

    def train(self, dataset, graficar=False):
        """
        Mueve los puntos centrales "centroides" a una distancia media 
        de los datos de cada dato del dataset.
        
        Args:
            - dataset: Un conjunto de datos de las caracteristicas que se evaluan en cada clase.
        """
        x ,y  = np.array(self.centroides).shape[:2]
        centroides_ant = np.random.rand(x, y)
        print("TRAINING...")
        diferencia = centroides_ant - self.centroides
        while diferencia.max() > 0:
            diferencia = centroides_ant - self.centroides
            centroides_ant = np.copy(self.centroides)
            distancias = []
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
        self.set_radios()

    def set_id_users(self, ids):
        self.ids = ids

    def that_class(self, entrada):
        """
        devuelve el nombre del grupo al que pertenece un punto de entrada.
        """
        if not len(self.centroides) == 0:
            distancias = np.linalg.norm(self.centroides-entrada, axis=1)
            minimo = distancias.min()
            indice = np.argmin(distancias)
            #print(minimo, '--', self.radios[indice])
            if minimo < self.radios[indice]: # debe de estar lo suficientemente cerca.
                return self.ids[indice]
            else:
                return None
        else:
            return None

    def add_class(self, id_user, codes):
        registered = False
        id_user = float(id_user)
        if codes.any():
            new_centroid = np.mean(codes, axis=0)
            if not self.dataset.any():
                self.dataset = codes
            else :
                self.dataset = np.append(self.dataset, codes, axis=0)
            if list(self.ids).count(id_user) >= 1:
                if self.that_class(new_centroid) == id_user:
                    self.train(self.dataset)
                    registered = True
                pass
            else:
                self.grupos += 1
                self.ids = np.append(self.ids, id_user)
                if not self.centroides.any():
                    self.centroides = np.array([new_centroid])
                    new_radio = np.linalg.norm(codes - new_centroid, axis=1).max()
                    self.radios.append(new_radio)
                else:
                    self.centroides = np.append(self.centroides, [new_centroid], axis=0)
                
                self.train(self.dataset)
                registered = True

            self.save_model()
        self.set_radios()
        return registered

    def set_radios(self):

        if self.grupos > 1:
            radios = []

            for v in self.centroides:
                centros = np.array(self.centroides).tolist()

                if centros.count(list(v)) >= 1:
                    centros.remove(list(v))
                centros = np.array(centros)
                radio = np.linalg.norm(centros - v, axis=1).min() * 0.75
                radios.append(radio)
            self.radios = radios
        elif self.grupos == 1:
            centro = self.centroides[0]
            radio = np.linalg.norm(self.dataset - centro, axis=1).max()
            self.radios.append(radio)
            
    
    def save_model(self):
        if not os.path.exists("dataset"):
            os.mkdir("dataset")
        np.savetxt(f"dataset/{self.model_name}_dataset.csv", self.dataset, fmt='%r', delimiter=',')
        database = np.concatenate([np.transpose([self.ids]), self.centroides], axis=1)
        np.savetxt(f'dataset/{self.model_name}_database.csv', database, fmt='%r', delimiter=',')
        
        # en caso de usar MongoDB
        
        # client = MongoClient()
        # db = client[self.model_name]
        # collections = db['users']
        # for index, id_user in enumerate(self.ids):
        #     new_user = {
        #                 '_id': id_user,
        #                 'code': self.centroides[index].tolist()
        #                 }
        #     if collections.find_one(id_user) == None:
        #         collections.insert_one(new_user).inserted_id
        #     else:
        #         collections.replace_one(collections.find_one(id_user), new_user)
        
    def load_model(self):

        if os.path.exists(f"dataset/{self.model_name}_dataset.csv"):
            self.dataset = np.loadtxt(f"dataset/{self.model_name}_dataset.csv", delimiter=',')
            if len(self.dataset.shape) < 2:
                self.dataset = np.array([self.dataset])
        if os.path.exists(f'dataset/{self.model_name}_database.csv'):
            database = np.loadtxt(f'dataset/{self.model_name}_database.csv', delimiter=',')
            if len(database.shape) < 2:
                database = np.array([database])
            ids = database[:,0:1]
            self.ids = np.ravel(np.int32(ids))
            self.centroides = database[:,1:]
            self.grupos = self.centroides.shape[0]
        self.set_radios()

        # en caso de usar MongoDB
        
        # client = MongoClient()
        # db = client[self.model_name]
        # collections = db['users']
        # for clase in collections.find():
        #     self.ids.append(clase['_id'])
        #     self.centroides.append(np.array(clase['code']))
        #     self.grupos += 1
        # 
        #print('Cargando kmeans...')