class Identifier:

    def __init__(self, face_encoder, k_mean, name = 'identificador'):

        self.face_encoder = face_encoder
        self.k_mean       = k_mean
        self.model_name   = name
        
    def identify(self, images):
        face_code = self.face_encoder(images)
        face_id = self.k_mean.that_class(face_code.mean(axis=0))
        return face_id
    
    def register(self, id_user, images):
        face_code = self.face_encoder(images)
        return self.k_mean.add_class(id_user, face_code)
    
    def __call__(self, images):
        return self.identify(images)
