# Preparación para investigar rocas espaciales mediante inteligencia artificial 👩‍🔬🦾

Conozca los aspectos sobre la investigación científica de rocas espaciales y cómo mejorarla con inteligencia artificial.

### Objetivos de aprendizaje

* Conocer los aspectos básicos de la inteligencia artificial
* Descubrir cómo los humanos clasifican objetos
* Descubrir cómo las máquinas clasifican objetos
* Conocer las bibliotecas de inteligencia artificial
* Instalar bibliotecas de inteligencia artificial

<hr/>

## Introducción

Imagine que trabaja como geólogo en la NASA. Su trabajo consiste en indicar a los astronautas qué rocas deben recoger en su viaje a la Luna. Como los astronautas tienen muchas otras cosas que aprender, no resulta práctico enseñarles sobre las distintas rocas de espacio. Así que decide crear un programa de inteligencia artificial (IA) que los astronautas podrán usar para hacer una fotografía a una roca y averiguar qué tipo de roca es. Al proporcionar un programa de inteligencia artificial a los astronautas, tendrá la garantía de que recogerán suficientes muestras de cada tipo de roca que necesita.

En este módulo, aprenderá qué es la inteligencia artificial y usará algunas bibliotecas que le ayudarán a crear un programa de clasificación de rocas. En el proyecto final se usa Visual Studio Code, Python y Jupyter Notebook para crear un modelo de inteligencia artificial que permita identificar el tipo roca a partir de una imagen.

Los datos se han obtenido de la NASA. Puede explorar fotografías más interesantes en la [colección de muestras](https://curator.jsc.nasa.gov/lunar/samplecatalog/index.cfm?azure-portal=true) de la NASA.

<hr/>

## Inteligencia artificial e investigación de rocas espaciales

La inteligencia artificial es una concentración bastante nueva pero importante en el campo de la ciencia informática y, más concretamente, en la ciencia de datos. El concepto principal de la inteligencia artificial es enseñar a una máquina a ser capaz de aprender cosas. Después, la máquina toma decisiones basadas en lo que ha aprendido.

Se podría pensar que enseñar a las máquinas a "pensar" por sí mismas podría conducir a la extinción de la raza humana. Pero no es necesario preocuparnos por que las máquinas puedan dominar el mundo. Necesitan a los humanos para programarlas. Nos beneficiamos de la programación de máquinas con inteligencia artificial. Estos son algunos ejemplos de cómo funciona la inteligencia artificial para mejorar nuestras vidas:

* Netflix usa algoritmos de inteligencia artificial para recomendar lo que nos gustaría ver a continuación.
* Siri ejecuta inteligencia artificial en los datos del teléfono para detectar patrones y ayudarnos a realizar tareas comunes más fácilmente.
* Tesla implementa inteligencia artificial para crear automóviles sin conductor de modo que todos podamos relajarnos y disfrutar de las vistas.

La inteligencia artificial consiste en proporcionar muchos datos diferentes a una máquina e indicarle qué significan. Para nuestro ejercicio, proporcionaremos a la máquina muchas imágenes de diferentes tipos de rocas espaciales y le informaremos sobre los tipos de roca. Cargaremos una foto de una roca de basalto e indicaremos lo siguiente a la máquina: *En esta foto, se muestra una roca de basalto*.

Este proceso es el primer paso para crear un *modelo de inteligencia artificial*. Usamos modelos de inteligencia artificial para realizar predicciones. Después de mostrar a la máquina un gran número de imágenes, crearemos un modelo a partir de los datos. A continuación, podemos proporcionar a la máquina una nueva foto y esta usará los datos del modelo para predecir cuál es el tipo de roca de la imagen.

Cuando use inteligencia artificial, puede que observe que los términos *aprendizaje automático* e *inteligencia artificial* se usan casi indistintamente. La principal diferencia entre los dos es que el aprendizaje automático es un tipo de inteligencia artificial. Son similares pero, en este módulo, nos centramos en la inteligencia artificial.

<hr/>

## Características y tipos de rocas espaciales

Empecemos aprendiendo cuáles son los distintos tipos de rocas espaciales que vamos a estudiar.

### Rocas lunares: disección de la Luna

En esta lección, usaremos dos tipos de rocas para clasificarlas: el *basalto* y la *roca de tierras altas* (también conocida como *cortical*). Ambos tipos de roca se encuentran en la Tierra. Pero, para nuestro estudio, solo veremos las variantes lunares, es decir, las rocas de la Luna.

#### Roca de basalto: excavación en un cráter lunar

El basalto es una roca oscura. Los científicos creen que procede de antiguas erupciones volcánicas en la Luna. Al observar la Luna, se aprecian zonas y manchas oscuras; es probable que lo que veamos sean rocas de basalto en la superficie lunar. Casi el 17 % del lado más cercano de la Luna está formado por basalto, y solo está presente en un 2 % del lado más lejano. La mayor parte del basalto de ambos hemisferios de la Luna se encuentra en cuencas o cráteres grandes.

![basalt-cristobalite-1](https://learn.microsoft.com/es-es/training/modules/introduction-artificial-intelligence-nasa/media/basalt-cristobalite-1.png)

![basalt-cristobalite-2](https://learn.microsoft.com/es-es/training/modules/introduction-artificial-intelligence-nasa/media/basalt-cristobalite-2.png)

#### Roca de tierras altas: barrido de la corteza lunar

La roca de tierras altas es más ligera que el basalto, porque este se compone de elementos más pesados, como hierro y magnesio. Una teoría sobre cómo se creó la roca de tierras altas es que un océano de magma grande cubrió la Luna cuando se formó y, a continuación, se cristalizó. Este tipo de roca es más ligero que el basalto. La teoría dice que flotó en la superficie del océano y pasó a convertirse en la corteza de la Luna.

![crustal-anorthosite-1](https://learn.microsoft.com/es-es/training/modules/introduction-artificial-intelligence-nasa/media/crustal-anorthosite-1.png)

![crustal-anorthosite-2](https://learn.microsoft.com/es-es/training/modules/introduction-artificial-intelligence-nasa/media/crustal-anorthosite-2.png)

<hr/>

## Clasificación de rocas espaciales como un humano

Para crear un modelo de inteligencia artificial que detecte tipos de roca, es necesario tener en cuenta la forma de clasificar objetos de los seres humanos.

En esta sección, describiremos un proceso de reflexión común que siguen los seres humanos para examinar y clasificar datos. Más adelante, usaremos estos pasos para formar un modelo con el fin de que nuestra máquina realice las mismas tareas.

### Paso 0: Obtención de los datos

Queremos recopilar tantas imágenes de rocas como sea posible. Si tenemos una gran cantidad de imágenes, podremos ver un gran número de variaciones en los elementos que intentamos clasificar. Por suerte, para este proyecto podemos elegir entre una gran cantidad de imágenes pertinentes que hay disponibles en Internet.

### Paso 1: Extracción de las características

En primer lugar, nuestro cerebro intenta extraer patrones de cada imagen. Los patrones incluyen estos factores:

* Combinaciones de colores
* Contornos afilados
* Patrones circulares
* Textura de la superficie de la roca
* Tamaño de la roca
* Tamaño y forma de los minerales en la roca

Nuestro cerebro realiza algunas de estas búsquedas y categorizaciones visuales de manera subconsciente. En inteligencia artificial, estos factores se denominan *características*.

En la siguiente imagen, se muestran algunas características que se pueden extraer de la fotografía de una motocicleta:

![ai-paso-1](https://learn.microsoft.com/es-es/training/modules/introduction-artificial-intelligence-nasa/media/features.png)

### Paso 2: Búsqueda de relaciones

Después, intentaremos encontrar la relación entre las características y el tipo de roca que se muestra en una fotografía de una roca.

En este paso, nuestro cerebro intenta separar o intercalar las características de cada tipo de roca. Debido a las asociaciones que hacemos, creamos algunas reglas. Determinamos que las rocas más ligeras suelen ser las rocas de tierras altas y que la textura de las rocas de basalto es más escalonada. Estas asociaciones y los vínculos que hay entre ellas se muestran en esta imagen:

![ai-paso-2](https://learn.microsoft.com/es-es/training/modules/introduction-artificial-intelligence-nasa/media/links.png)

### Paso 3: Clasificación de los tipos

Por último, intentamos usar estas relaciones entre elementos conocidos para determinar cómo clasificar un nuevo elemento. Cuando encontramos una nueva imagen de una roca, nuestro cerebro extrae sus características. Después, usamos las asociaciones que ya hemos realizado para determinar qué tipo de roca es.

![ai-paso-3](https://learn.microsoft.com/es-es/training/modules/introduction-artificial-intelligence-nasa/media/association-process.png)

<hr/>

## Clasificación de rocas espaciales mediante inteligencia artificial

El trabajo de los científicos de inteligencia artificial al crear un modelo de este tipo consiste en *enseñar* a la máquina a alcanzar el objetivo. Para la investigación de rocas espaciales, el objetivo es que el sistema de clasificación tenga una precisión del 100%. El 100% parece imposible para los seres humanos. Sin embargo, cuando los científicos integran las máquinas y la inteligencia artificial con sus otras técnicas de investigación, el objetivo está a su alcance.

Con un modelo de inteligencia artificial, a menudo se implementan los mismos pasos del proceso humano, o unos similares, para lograr el objetivo. El científico *enseña* a la máquina mediante la creación del modelo. Esta *aprende* repitiendo el proceso del modelo. Cada iteración del modelo produce más datos. Cuantos más datos se recopilen y analicen, más precisión ofrecerá la máquina al realizar las predicciones.

Para nuestro modelo de inteligencia artificial, comenzaremos con los pasos que realizaría un humano para examinar y clasificar rocas. Enseñaremos a la máquina a seguir estos pasos. Una vez que esta ejecute el modelo y genere datos de análisis, podrá predecir con precisión el tipo de roca a partir de datos nuevos.

### Paso 0: Obtención de los datos

Un paso de preparación consiste en importar datos de imagen. También necesitamos obtener las bibliotecas para ayudar a procesar los datos en la máquina que entrenaremos. La máquina transformará las imágenes en matrices de números para que estén en un formato que pueda leer.

### Paso 1: Extracción de las características

A partir de las fotos de rocas (datos) que proporcionamos, el equipo extraerá características como la textura, el tamaño, el color y los bordes. Los científicos usan la intuición y la experiencia para especificar las características que se deben buscar.

### Paso 2: Búsqueda de asociaciones

El equipo realiza asociaciones entre características de la imagen y los tipos de roca. Las máquinas pueden ser mejores que los humanos en la tarea de detectar detalles sutiles, ya que hay muchas asociaciones que realizar.

La máquina creará una red que podrá realizar un seguimiento de millones de asociaciones.

### Paso 3: Predicción de los tipos

La máquina extraerá las características de roca definidas de una nueva foto. Usará asociaciones entre los datos existentes y los datos de la nueva foto para predecir de qué tipo de roca se trata.

## Bibliotecas de Python comunes para proyectos de inteligencia artificial

Más adelante en esta ruta de aprendizaje, usaremos tres bibliotecas de Python:

* Matplotlib
* NumPy
* PyTorch

Las bibliotecas son gratuitas y se suelen usar en casos reales de proyectos de inteligencia artificial.

### Matplotlib

La biblioteca Matplotlib se usa principalmente para visualizar datos en Python. Matplotlib sirve para crear visualizaciones estáticas, animadas e interactivas en Python. Matplotlib es útil para mostrar los datos de una manera más visual.

### NumPy

La biblioteca NumPy, que es la abreviatura de *Numerical Python*, es una opción muy popular que se usa para organizar y almacenar datos en Python. Puede usar NumPy para crear estructuras que contengan conjuntos de datos denominados *matrices*. Al igual que las listas, las matrices almacenan muchos tipos de datos. NumPy tiene muchas funciones que son útiles para manipular datos en matrices.

### PyTorch

La biblioteca PyTorch es una biblioteca de aprendizaje automático. Tiene muchas funciones integradas que ayudan a acelerar la compilación de proyectos. PyTorch se usa principalmente para modificar los datos de un programa de aprendizaje automático existente.

<hr/>

## Ejercicio: Descarga de bibliotecas inteligencia artificial de Python

En esta sección, instalaremos las bibliotecas que necesitará para crear el modelo de inteligencia artificial a medida que continuamos con la ruta de aprendizaje. Usaremos Anaconda para completar las descargas. [Anaconda](https://www.anaconda.com/about-us/) es una distribución de los lenguajes de programación [Python](https://www.python.org/about/) y [R](https://www.r-project.org/about.html). Incluye bibliotecas para el desarrollo en la ciencia computacional, como ciencia de datos, aprendizaje automático, análisis predictivo, etc.

### Descargar Anaconda

1. Vaya a la [página de descarga de Anaconda](https://www.anaconda.com/download/) para instalar esta distribución.
2. Seleccione **Descargar**.
3. En la lista de vínculos de descarga, elija el vínculo que corresponda al sistema operativo de su equipo. Espere a que termine la descarga.
4. Para iniciar la instalación, seleccione el archivo ejecutable en la esquina inferior izquierda del explorador. También puede abrir la carpeta de descargas y ejecutar el archivo ejecutable desde esa ubicación.
5. Siga los pasos para instalar Anaconda en el equipo.

Una vez completada la instalación, la aplicación estará disponible en el equipo:

* Si usa Windows, ejecute el **símbolo del sistema de Anaconda** desde el menú Inicio.
* En un equipo Mac, ejecute el **símbolo del sistema de Anaconda** en el terminal.

### Uso de Anaconda para instalar bibliotecas de inteligencia artificial

1. En el símbolo del sistema de Anaconda, ejecute el comando `conda create` para iniciar el entorno de Anaconda:

    ```
    conda create -n myenv python=3.XX pandas jupyter seaborn scikit-learn keras pytorch pillow
    ```

    Este comando usa Anaconda para instalar todas las bibliotecas que necesitamos para nuestro modelo. Con este comando, también descargaremos algunas bibliotecas adecuadas para la ciencia de datos. Es posible que estas bibliotecas le resulten útiles para el desarrollo en el futuro.

2. Cuando se le pida que instale los paquetes, escriba **Y** y presione Enter.

3. Para activar el nuevo entorno, ejecute el comando `conda activate`:

    ```
    conda activate myenv
    ```

El nuevo entorno estará listo para usarse, pero necesitaremos agregar una biblioteca más a través de un comando de instalación independiente.

### Instalación del paquete de torchvision

1. En el símbolo del sistema de Anaconda, ejecute el comando `conda install`.

    ```
    conda install -c pytorch torchvision
    ```

2. Cuando se le pida que instale el paquete, escriba **Y** y presione Enter.

### Creación de una carpeta de proyecto y un archivo de Jupyter Notebook

Ahora tiene un entorno que puede usar para el resto de la ruta de aprendizaje. El último paso es crear una carpeta de proyecto para los archivos de código fuente.

1. Elija una ubicación de fácil acceso en el equipo y cree una carpeta denominada **ClassifySpaceRocks**.

2. Abra Visual Studio Code y abra la carpeta que ha creado.

3. Cree un archivo de Jupyter Notebook denominado **ClassifySpaceRockProgram**.
    - Presione `CTRL + Mayús + P` para abrir el menú desplegable **Comando** situado en la parte superior de Visual Studio.
    - Seleccione **Jupyter: Create new blank notebook** (Jupyter: Crear cuaderno en blanco).
    Se abrirá un nuevo cuaderno. El sistema mostrará un mensaje sobre cómo conectarse al kernel de Python en la esquina inferior derecha.
    - Agregue el siguiente comentario en la primera celda del nuevo cuaderno:

        ```python
        # AI model to classify space rocks
        ```

    - Seleccione la flecha verde de la parte superior de la celda para ejecutarlo.
    - Presione `CTRL + S` para guardar el archivo.
    - En el cuadro de diálogo **Guardar como**, vaya a su carpeta.
    - Escriba el nombre del nuevo cuaderno. En nuestro ejemplo, usaremos el nombre **ClassifySpaceRockProgram**. Asegúrese de que **Jupyter** esté seleccionado como   tipo de archivo.
    - Seleccione **Guardar**.
    El archivo de Jupyter Notebook guardado deberá tener la extensión *.ipynb*. Debería ver el archivo en la vista **Explorador** de Visual Studio.

4. En las esquinas superior derecha e inferior izquierda de Visual Studio, cambie el entorno al nuevo entorno de Anaconda que ha creado.

    ![set-environment](https://learn.microsoft.com/es-es/training/modules/introduction-artificial-intelligence-nasa/media/set-environment.png)