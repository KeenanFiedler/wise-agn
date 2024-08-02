import numpy as np
import h5py as h5
import math
import matplotlib.pyplot as plt

from keras.models import load_model

path = "C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\"
wave = [0.01,0.015,0.022,0.033,0.049,0.073,0.1,0.123,0.151,0.185,0.227,0.279,0.343,0.421,0.517,
        0.55,0.635,0.78,0.958,1.18,1.45,1.77,2.18,2.68,3.29,4.04,4.96,6,6.25,6.5,6.75,7,7.25,7.5,
        7.75,8,8.25,8.5,8.6,8.7,8.8,8.9,9,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10,10.1,10.2,10.3,10.4,
        10.5,10.6,10.7,10.8,10.9,11,11.1,11.2,11.3,11.4,11.5,11.8,12,12.3,12.5,12.8,13,13.3,13.5,13.8,
        14,14.3,14.5,14.8,15,15.3,15.5,15.8,16,16.3,16.5,16.8,17,17.3,17.5,17.8,18,18.3,18.5,18.8,19,
        19.3,19.5,19.8,20,20.9,25.6,31.5,38.7,47.5,58.3,71.6,87.9,108,133,163,200,204,304,452,672,1000]



model = load_model(path + 'autoencoder\\3layer_64\\model_decoder_gpu_64.keras')

def generate_seds(model, n_sed, min_array = [5.0,1.0,0.0,0.0,15.0,10.0], max_array = [100.0,15.0,1.0,3.0,70.0,300.0]):
    # Draw random values in between the minimum and maximum parameters, of size requested
    random_draws = np.random.uniform(min_array, max_array, size=(n_sed,6))
    # load machine learning model and find SEDs
    fluxes = 10**model.predict(random_draws)

    return fluxes

seds = generate_seds(model, 10)

for s in seds:
    plt.loglog(wave, s)

plt.show()
plt.clf()