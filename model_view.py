from face_identity.kmean import K_mean

k_model = K_mean('scesi_auth_km')

k_model.train(k_model.dataset)