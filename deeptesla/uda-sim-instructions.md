## Preparando el simulador del autopilot
* Descargar e instalar el simulador desde https://github.com/udacity/self-driving-car-sim
    * En la sección _Avaliable Game Builds_ se hallan disponibles binarios precompilados para su descarga. Nota: todavía no se si se necesita Unity si se instala el simulador por este medio.

* Descargar e instalar Anaconda desde https://www.anaconda.com/download/
* Clonar el proyecto:
```sh
git clone https://github.com/naokishibuya/car-behavioral-cloning.git
```
* Crear el entorno virtual con conda
```sh
cd car-behavioral-cloning
conda env create -f environments.xml
```
* Activar el entorno y actualizar keras
```sh
source activate car-behavioral-cloning
pip install --upgrade keras
```
* En utils.py cambiar:
```py
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
```
```py
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 64, 128, 3
```
* En drive.py cambiar:
1)
```py
image = np.array([image])       # the model expects 4D array
```
```py
image = np.array([image], dtype = 'float32')
image /= 255
```
2)
```py
steering_angle = float(model.predict(image, batch_size=1))
```
```py
STEERING_MAX = 25
steering_angle = float(model.predict(image, batch_size=1)) / STEERING_MAX
```
* Probar
```py
python drive.py deeptesla_trained_model.h5
```