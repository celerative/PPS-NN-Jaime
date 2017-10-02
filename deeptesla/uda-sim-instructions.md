## Setting up the autopilot simulator
* Download and install the simulator from https://github.com/udacity/self-driving-car-sim
    * At section _Avaliable Game Builds_ there are precompiled binaries available for download.
    **Note:** If the simulator is installed this way, I don't know if you must install Unity to make it work properly.
* Download and install Anaconda from https://www.anaconda.com/download/
* Clone this project:
```sh
git clone https://github.com/naokishibuya/car-behavioral-cloning.git
```
* Create the virtual environment using conda.
```sh
cd car-behavioral-cloning
conda env create -f environments.xml
```
* Activate the environment and update keras.
```sh
source activate car-behavioral-cloning
pip install --upgrade keras
```
* At utils.py, change:
```py
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
```
```py
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 64, 128, 3
```
* At drive.py, change:
**Note:** There is no need anymore to normalize the input and output here due to that operation has been built on the CNN and its training.
```py
image = np.array([image])       # the model expects 4D array
```
```py
image = np.array([image], dtype = 'float32')
#image /= 255
```
* Try it out.
```py
python drive.py deeptesla_trained_model.h5
```