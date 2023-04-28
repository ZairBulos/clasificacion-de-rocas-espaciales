# Análisis de imágenes de rocas mediante inteligencia artificial 👨‍💻🔬

Identifique los datos que debe agregar a un módulo de inteligencia artificial que clasifica las rocas espaciales de fotos aleatorias.

### Objetivos de aprendizaje

* Importar bibliotecas de inteligencia artificial
* Descargar e importar datos para usarlos con un programa de inteligencia artificial
* Aprender a limpiar y separar datos
* Descubrir cómo los equipos leen fotos como imágenes mediante el formato binario
* Usar código para leer una imagen y asignar el tipo de roca correcto

<hr/>

## Introducción

En los dos primeros módulos, ha obtenido información sobre la inteligencia artificial (IA) y cómo se puede usar la tecnología para mejorar un proyecto. Ahora puede empezar a crear un programa propio que use inteligencia artificial.

La instalación de las bibliotecas de inteligencia artificial de Python es un primer paso excelente para crear un programa de clasificación de rocas. Las bibliotecas ofrecen un modelo base que puede entrenar para completar la tarea a mano.

En este módulo, usaremos Visual Studio Code, Python y Jupyter Notebook. Revisará código para limpiar imágenes de rocas para preparar los datos para el programa. A continuación, veremos el código que utilizaremos para crear y entrenar un modelo de inteligencia artificial.

<hr/>

## Ejercicio: Importación de bibliotecas de Python en Jupyter Notebook

Ahora que ha descargado las bibliotecas, puede empezar a importarlas en un archivo de Jupyter Notebook.

### Adición de instrucciones de importación para las bibliotecas

Siga estos pasos para agregar código para importar las bibliotecas de inteligencia artificial. Inserte cada nueva sección de código en una celda vacía del archivo de Jupyter Notebook. Seleccione la flecha verde en la parte superior de la celda para ejecutar el nuevo código.

1. Abra Visual Studio Code y, a continuación, abra el archivo de Jupyter Notebook que creó en el módulo anterior.
    En el módulo anterior, se le asignó el nombre *ClassifySpaceRockProgram.ipynb* al archivo de Jupyter Notebook.

2. Asegúrese de que ejecuta el kernel de Jupyter correcto. En las esquinas superior derecha e inferior izquierda de Visual Studio, cambie al entorno de Anaconda (`'myenv'`) que creó en el último módulo.

3. La primera biblioteca que se va a importar es **Matplotlib**. Esta biblioteca se usa para trazar los datos. Agregue el código siguiente en una nueva celda del archivo de Jupyter Notebook y, a continuación, ejecute el código.

    ```python
    import matplotlib.pyplot as plt
    ```

    Asegúrese de que la instrucción no comienza con un símbolo de almohadilla (#). De lo contrario, Python interpretará la instrucción como un comentario.

4. A continuación, agregue el código siguiente para importar la biblioteca **NumPy**, que se va a usar para procesar matrices numéricas (imágenes) de gran tamaño, y ejecute la nueva celda.

    ```python
    import numpy as np
    ```

5. Ahora agregue código en una nueva celda para importar la biblioteca **PyTorch**, que se va a usar para entrenar y procesar modelos de aprendizaje profundo e inteligencia artificial. Después de agregar el nuevo código, ejecute la celda.

    ```python
    import torch
    from torch import nn, optim
    from torch.autograd import Variable
    import torch.nn.functional as F
    ```

6. La siguiente biblioteca que se va a importar es **torchvision**, que forma parte de **PyTorch**. Esta biblioteca se usa para procesar imágenes y realizar manipulaciones como recortar y cambiar el tamaño. Agregue este código en una nueva celda para importar la biblioteca y, a continuación, ejecute la celda.

    ```python
    import torchvision
    from torchvision import datasets, transforms, models
    ```

7. Ahora agregue código para importar **Python Imaging Library** (PIL) para poder visualizar las imágenes. Después de agregar el nuevo código, ejecute la celda.

    ```python
   from PIL import Image
    ```

8. Por último, agregue el código siguiente para importar dos bibliotecas que garantizan que los trazados se muestren insertados y en alta resolución. Después de agregar el nuevo código, ejecute la celda.

    ```python
    %matplotlib inline
    %config InlineBackend.figure_format = 'retina'
    ```

<hr/>

## Cómo limpiar y separar los datos para los proyectos de inteligencia artificial

El siguiente paso es importar los datos de las imágenes de rocas existentes que usaremos para enseñar a nuestro equipo a reconocer los distintos tipos de rocas.

Antes de importar imágenes, es necesario revisar dos pasos críticos del proceso de inteligencia artificial: la limpieza y separación de datos. Es importante completar estos pasos para asegurarse de que el equipo pueda clasificar con precisión las imágenes de rocas.

### Limpiar datos

Para limpiar los datos, tenemos que asegurarnos de que están completos y son uniformes. En nuestro ejemplo de rocas, muchos de los archivos de imagen tienen distintos tamaños. Para limpiar este conjunto, debemos cambiar el tamaño de cada archivo de imagen para que todos tengan el mismo. Es posible que tengamos que rellenar celdas en las que faltan datos y eliminar filas con datos incorrectos.

### Separación de datos

Para programar la inteligencia artificial, primero debemos proporcionar al equipo muchos datos y decirle qué representan. Este proceso se denomina *entrenamiento*. Después de entrenar el equipo, se *prueba* para ver si puede clasificar los nuevos datos que se le proporcionen.

La NASA ha proporcionado una gran cantidad de datos sobre los diferentes tipos de roca. Es necesario indicar al equipo qué datos se usarán para entrenamiento y cuáles se usarán para pruebas. Para realizar la separación, distribuimos los datos de forma aleatoria en estos dos grupos. La proporción de la cantidad de datos que se coloca en cada grupo puede variar. En nuestro ejemplo, haremos el entrenamiento con el 80% de los datos y la prueba con un 20% de los datos.

<hr/>

## Ejercicio: Importación y limpieza de datos de fotos

Ahora que ya sabemos cómo limpiar y separar los datos, ya podemos aplicar estos principios a nuestro proyecto de clasificación de rocas.
 
### Preparación de los datos

Es necesario crear dos conjuntos de datos a partir de las fotos de la NASA para nuestro proyecto de clasificación. Un conjunto de datos es para entrenamiento y el otro para pruebas. Las imágenes deben limpiarse y separarse antes de cargarlas en conjuntos de datos para su procesamiento. Los datos deben procesarse de forma aleatoria y no en el orden exacto en que los proporcionó la NASA.

Usaremos código para realizar estos cuatro pasos para preparar los datos:

* Paso 1. **Obtener los datos**: dígale al equipo de dónde obtener los datos de la imagen.
* Paso 2. **Limpiar los datos**: recorte las imágenes para que tengan el mismo tamaño.
* Paso 3. **Separar los datos**: separe los datos mediante la ordenación y la selección aleatoria.
* Paso 4. **Cargar conjuntos de datos aleatorios**: prepare ejemplos aleatorios para los conjuntos de datos de entrenamiento y de pruebas.

#### Paso 1. Obtener los datos

Es necesario que el equipo sepa dónde puede encontrar los datos. En nuestro ejemplo, usamos las imágenes de rocas proporcionadas por la NASA. Ya hemos descargado y almacenado las fotos en la carpeta *Data* que se encuentra en la misma carpeta de proyecto que el archivo de Jupyter Notebook. Le diremos al equipo que cargue los datos de la imagen desde la carpeta de *Data*.

#### Paso 2. Limpiar los datos

Las fotos de rocas de la NASA tienen diferentes tamaños: pequeño, mediano y grande. Recortaremos las imágenes para que tengan el mismo tamaño (224 x 224 píxeles). Cambiamos el tamaño de las imágenes porque los equipos esperan que las imágenes tengan el mismo tamaño. Si las imágenes tienen diferentes tamaños, no es tan fácil que el equipo las procese. Usamos la clase `transforms.Compose` de torchvision para cambiar el tamaño de las imágenes a las dimensiones que prefiramos y almacenar las imágenes modificadas en variables locales.

#### Paso 3. Separar los datos

El 20% de las imágenes limpiadas son para entrenamiento y el otro 80% para pruebas. El equipo debe elegir imágenes de forma aleatoria y no usarlas en el orden exacto en que las proporcionó la NASA. Usamos dos técnicas para realizar la separación: la ordenación y la selección aleatoria.

Creamos una lista de índices que corresponde al número de imágenes. Usamos esta lista para buscar el índice de la imagen que representa el 20% de los datos. Almacenamos esta ubicación en una variable denominada `split`. Ordenamos de forma aleatoria la lista de índices y, mediante la ubicación de la imagen en `split`, creamos los dos conjuntos de datos para entrenamiento y pruebas. Los conjuntos resultantes constan de imágenes que se limpian y se seleccionan de forma aleatoria.

Usamos la función `load_split_train_test` para obtener los datos ordenados de forma aleatoria para el entrenamiento y las pruebas.

#### Paso 4. Cargar conjuntos de datos aleatorios

Para cargar imágenes aleatorias de los dos conjuntos de datos, llamamos a la función `SubsetRandomSampler` desde la biblioteca torch.utils.data.sampler. Cargaremos muestras aleatorias de 16 imágenes cada una.

### Adición de código para limpiar y separar los datos

Estamos listos para agregar el código para limpiar y separar los datos.

1. Descargue la carpeta *Data.zip*.
2. Descomprima la carpeta de Datos y colóquela en la misma carpeta que el archivo de Jupyter Notebook.
3. En Visual Studio Code, vuelva al archivo de Jupyter Notebook.
4. Agregue el código siguiente en una nueva celda para importar **Python Imaging Library** (PIL). Usaremos esta biblioteca para visualizar las imágenes. Después de agregar el nuevo código, ejecute la celda.

    ```python
    # Tell the machine what folder contains the image data
    data_dir = './Data'

    # Read the data, crop and resize the images, split data into two groups: test and train
    def load_split_train_test(data_dir, valid_size = .2):

    # Transform the images to train the model
    train_transforms = transforms.Compose([
                                       transforms.RandomResizedCrop(224),
                                       transforms.Resize(224),
                                       transforms.ToTensor(),
                                       ])

    # Transform the images to test the model
    test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.Resize(224),
                                          transforms.ToTensor(),
                                      ])

    # Create two variables for the folders with the training and testing images
    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir, transform=test_transforms)

    # Get the number of images in the training folder
    num_train = len(train_data)

    # Create a list of numbers from 0 to the number of training images - 1
    # Example: For 10 images, the variable is the list [0,1,2,3,4,5,6,7,8,9]
    indices = list(range(num_train))

    # If valid_size is .2, find the index of the image that represents 20% of the data
    # If there are 10 images, a split would result in 2
    # split = int(np.floor(.2 * 10)) -> int(np.floor(2)) -> int(2) -> 2
    split = int(np.floor(valid_size * num_train))

    # Randomly shuffle the indices
    # For 10 images, an example would be that indices is now the list [2,5,4,6,7,1,3,0,9,8]
    np.random.shuffle(indices)

    from torch.utils.data.sampler import SubsetRandomSampler

    # With the indices randomly shuffled, 
    # grab the first 20% of the shuffled indices, and store them in the training index list
    # grab the remainder of the shuffled indices, and store them in the testing index list
    # Given our example so far, this would result is:
    # train_idx is the list [1,5] 
    # test_idx is the list [4,6,7,1,3,0,9,8]
    train_idx, test_idx = indices[split:], indices[:split]

    # Create samplers to randomly grab items from the training and testing indices lists
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Create loaders to load 16 images from the train and test data folders
    # Images are chosen based on the shuffled index lists and by using the samplers
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=16)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=16)

    # Return the loaders so you can grab images randomly from the training and testing data folders
    return trainloader, testloader

    # Using the function that shuffles images,
    # create a trainloader to load 20% of the images
    # create a testloader to load 80% of the images
    trainloader, testloader = load_split_train_test(data_dir, .2)

    # Print the type of rocks that are included in the trainloader
    print(trainloader.dataset.classes)
    ```

Después de ejecutar la celda, debería ver los dos tipos de clasificación de rocas en la salida: `['Basalt', 'Highland']`.

Los datos de las rocas espaciales ya se han importado, limpiado y separado. Estamos listos para entrenar nuestro modelo con el 80% de los datos y ejecutar pruebas con el 20% restante.

<hr/>

## Procedimiento que siguen los equipos para leer las fotos como archivos de imagen

Ahora que hemos limpiado y separado nuestros datos, es posible que se pregunte cómo lee la máquina estas imágenes.

> *Pista: los equipos no pueden leer imágenes de la misma manera que lo hacen los seres humanos.*

Si sabe algo sobre desarrollo informático, probablemente sabe que los equipos leen datos en formato *binario*. Los datos se representan como una serie larga de unos y ceros como 101011001110001010111, y así sucesivamente.

Por lo tanto, ¿cómo puede un equipo leer una imagen compleja solo como una serie de unos y ceros?

Si amplía los datos de una imagen, verá que la imagen de la foto se representa en el archivo de imagen como *píxeles*. Cada píxel es un color específico que tiene un código único. Una vez que el equipo convierte una foto en una imagen con estos códigos, puede leer y descifrar los datos de píxeles binarios.

Este es un ejemplo que muestra cómo un equipo transforma una foto en una serie de números en un archivo de imagen:

![lincoln](https://learn.microsoft.com/es-es/training/modules/analyze-rock-images-ai-nasa/media/lincoln.png)

![pixels](https://learn.microsoft.com/es-es/training/modules/analyze-rock-images-ai-nasa/media/pixels.png)

<hr/>

## Ejercicio: Presentación de fotos en Jupyter Notebook

Ahora veremos algunas de las imágenes que hemos cargado en el equipo. Les asignaremos etiquetas que indiquen el tipo de roca que hay en cada foto.

### Transformación y visualización de las imágenes

En esta sección, agregaremos código para hacer coincidir cada imagen de roca con un tipo de roca, en función de la carpeta de imágenes. Llamamos de nuevo a la clase `transforms.Compose` para transformar cada imagen en píxeles y cambiar su tamaño a las dimensiones que prefiramos.

Seleccionamos un conjunto de imágenes aleatoriamente de forma similar a como se usaron las funciones `load_split_train_test` y `SubsetRandomSampler` en el último ejercicio. El código recorre en iteración las imágenes ordenadas de forma aleatoria del conjunto de datos de pruebas.

La última sección de código muestra las imágenes que se cargan en el programa. Usamos funciones de la biblioteca PIL para manipular las imágenes y el comando `plt.show` para imprimirlas.

### Adición de código para transformar y seleccionar imágenes aleatorias

Estamos listos para agregar el código para transformar las imágenes.

1. En Visual Studio Code, vuelva al archivo de Jupyter Notebook.

2. Agregue el código siguiente en una nueva celda. Después de agregar el nuevo código, ejecute la celda.

    ```python
    # Transform an image into pixels and resize it
    test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                   transforms.Resize(224),
                                   transforms.ToTensor(),
                                 ])

    # Randomly select a set of images by using a similar approach as the load_split_train_test function
    def get_random_images(num):
        data = datasets.ImageFolder(data_dir, transform=test_transforms)
        classes = data.classes
        indices = list(range(len(data)))
        np.random.shuffle(indices)
        idx = indices[:num]
        from torch.utils.data.sampler import SubsetRandomSampler
        sampler = SubsetRandomSampler(idx)
        loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)

        # Create an iterator to iterate over the shuffled images in the test image dataset
        dataiter = iter(loader)

        # Get and return the images and labels from the iterator
        images, labels = dataiter.next()
        return images, labels
    ```

### Adición de código para mostrar imágenes seleccionadas aleatoriamente

Siga estos pasos para agregar el código para mostrar las imágenes.

1. Agregue el código siguiente en una nueva celda. Después de agregar el nuevo código, ejecute la celda.

    ```python
    # Show five images - you can change this number
    images, labels = get_random_images(5)

    # Convert the array of pixels to an image
    to_pil = transforms.ToPILImage()
    fig = plt.figure(figsize=(20,20))

    # Get a list of all classes in the training data
    classes=trainloader.dataset.classes

    # Draw the images in a plot to display in the notebook
    for ii in range(len(images)):
        image = to_pil(images[ii])
        sub = fig.add_subplot(1, len(images), ii+1)
        plt.axis('off')
        plt.imshow(image)

    # Display all of the images 
    plt.show()
    ```

2. Presione `Ctrl + S` para guardar los cambios en el archivo de Jupyter Notebook.

Después de ejecutar este nuevo código, debería ver cinco imágenes limpiadas en la salida. El código está configurado para mostrar cinco imágenes, pero puede cambiar el número.