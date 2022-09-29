import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = np.loadtxt('datasets/NoiseArrayTestNoise.txt', delimiter=' ')  # Test empty array
#dataset = np.loadtxt('datasets/NoiseArrayTestQuake.txt', delimiter=' ') # Test array with quake

class EarthquakeDetector:

    def __init__(self, 
                seismograph_count=5, 
                sample_rate_hz=3000, 
                alert_callback=False):

        json_file = open('model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        seismic_model = tf.keras.models.model_from_json(loaded_model_json)
        seismic_model.load_weights("model/model.h5")

        self.seismograph_count = seismograph_count
        self.alert_callback = alert_callback
        self.b = np.zeros((seismograph_count, sample_rate_hz, 1), dtype=int)  

        for x in range(seismograph_count):
            self.b[x] = (seismic_model.predict(dataset) > 0.5).astype(int)

    pass

    def new_samples(self, seismograph_id, samples):

        fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True)
        
        ax.plot(samples[:,0], linestyle="-", marker=".", lw=1, markersize=1)
        ax.set_title("Dataset")

        ax2.plot(samples[:,0], linestyle="-", marker=".", lw=1, markersize=1)
        ax2.plot(self.b[seismograph_id], color="r", lw=10, markersize=10, alpha=0.5)
        ax2.set_title("Earthquake Detected")

        specificSum = np.sum(self.b[seismograph_id])
        netSum = 0

        for y in range(self.seismograph_count):
            netSum = netSum + np.sum(self.b[y])
        
        netSum = netSum / self.seismograph_count

        if (netSum - 100) <= specificSum <= (netSum + 100):
            self.alert_callback = True

        plt.show()

    pass

if __name__ == "__main__":

    ed = EarthquakeDetector()
    ed.new_samples(seismograph_id=0, samples=dataset)


    if (ed.alert_callback == True):
        print("Earthquake confirmed by all seisemographs!")
    else:
        print("No Earthquake could be confirmed")

        