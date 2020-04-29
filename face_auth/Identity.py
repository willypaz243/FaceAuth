

class Identity:

    def __init__(self, frame_box, id_entity):

        self.__frame_box = frame_box
        self.__id_entity = id_entity
        
    def __str__(self):
        return str(self.__frame_box) + "\nID: " + self.__id_entity
