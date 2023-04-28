# Clasificación de tipos de rocas espaciales en fotos aleatorias mediante inteligencia artificial 💻🔢

Aprenda a crear un modelo de inteligencia artificial para predecir los tipos de rocas espaciales en las imágenes. Entrene y pruebe el modelo mediante fotos aleatorias.

### Objetivos de aprendizaje

* Entrenar un modelo de inteligencia artificial
* Probar el modelo usándolo para clasificar fotos aleatorias de rocas espaciales

<hr/>

## Introducción

En este momento, tiene los datos importados, limpios y listos para el entrenamiento y las pruebas. Ahora puede crear y entrenar un modelo de inteligencia artificial (IA) mediante los datos.

Las bibliotecas que importó realizarán la mayor parte del trabajo pesado automáticamente. El trabajo es indicar al equipo cómo entrenar el modelo para que pueda realizar predicciones.

En este módulo, creará un modelo de IA con una red neuronal. El modelo identificará el tipo de roca espacial en una imagen. Usaremos Visual Studio Code, Python y Jupyter Notebook para crear, entrenar y probar el modelo.

<hr/>

## Extracción de características de una imagen para el procesamiento de inteligencia artificial

Hemos limpiado y separado los datos de nuestro programa. Ahora estamos listos para entrenar el equipo para reconocer las diferentes características de los tipos de rocas espaciales.

Para entrenar el equipo, debe extraer características de las imágenes. Este paso puede parecer poco intuitivo. Nuestros cerebros extraen automáticamente las características de las imágenes, por lo que normalmente no lo notamos.

Hemos aprendido que cada imagen es una colección de píxeles, los cuales se representan mediante números. Para entrenar nuestro modelo, revisaremos cada imagen como una matriz de números.

Para extraer las características de una imagen, multiplicamos la imagen por filtros. Cada filtro se usa para extraer una característica determinada.

En la foto siguiente, vemos cómo se mueven los filtros sobre una imagen para extraer características como los bordes, las curvas y la textura de una roca.

Usaremos 32 filtros para clasificar las rocas de nuestro modelo, pero hay más filtros disponibles.

![filters](https://learn.microsoft.com/es-es/training/modules/train-test-predictive-ai-model-nasa/media/filters.gif)

> *Crédito de visualización*: Grant Sanderson, [https://www.3blue1brown.com/.](https://www.3blue1brown.com/)

<hr/>

## Ejercicio: Creación de una red neuronal para la clasificación de rocas espaciales

Crearemos una red neuronal (o red de aprendizaje profundo) para aprender las asociaciones entre las características y cada tipo de roca. Las características pueden incluir elementos como curvas, bordes y textura.

### Neuronas y redes cableadas

Las redes neuronales procesan información de forma similar al funcionamiento de nuestros cerebros. Nuestros cerebros constan de neuronas o células nerviosas que transmiten y procesan la información que recibe de los sentidos. Muchas células nerviosas se organizan como una red de nervios en el cerebro. Los nervios pasan los impulsos eléctricos de una neurona a la siguiente en la red.

Las redes neuronales tienen millones de neuronas y nervios y, para crear una red neuronal funcional, conectamos las neuronas y los nervios entre sí en dos pasos:

* Paso A: Creación de todas las neuronas.
* Paso B: Conexión de las neuronas de forma **adecuada** (existen miles de formas de conectar neuronas).

En nuestro modelo, recopilaremos las características de una roca de una imagen y las almacenaremos como una secuencia lineal de parámetros. Este paso crea una única neurona. Cada nueva imagen que se analiza es otra neurona. Proporcionamos los datos de entrenamiento para que nuestro equipo compile todas las neuronas.

A continuación, indicaremos al equipo que combine las secuencias en una matriz. La matriz representa el mejor patrón que tenemos para describir las características de los tipos de roca espacial. Esta matriz es una red cableada.

Entrenaremos nuestro modelo para predecir el tipo de roca. Compararemos las características de la roca de una nueva imagen con el patrón de matriz. Cada vez que ejecutamos el modelo, la matriz crece y mejora la precisión de la predicción. Nuestro objetivo es probar el modelo y lograr una precisión de la predicción cercana al 100%.

### Comprobación del entorno de trabajo

Para poder agregar código nuevo al modelo de IA, es necesario asegurarse de que el entorno de desarrollo sigue activo.

Si cerró el símbolo del sistema de Anaconda o Visual Studio Code, debe reiniciarlos. Deberá configurar el entorno para seguir trabajando en el modelo de IA.

Si el símbolo del sistema de Anaconda sigue abierto desde el trabajo del módulo anterior y no ha cerrado Visual Studio Code, continúe con la sección, [Creación de una red neuronal]().

#### Comprobación del entorno de Anaconda (myenv)

Si cerró el símbolo del sistema de Anaconda después de completar los ejercicios del módulo anterior, siga estos pasos para reiniciar el entorno.

1. Inicie la aplicación Anaconda prompt (o terminal en Mac).

2. En el símbolo del sistema de Anaconda, escriba el siguiente comando para activar el entorno:

    ```
    conda activate myenv
    ```
3. Use el siguiente comando para comprobar la instalación del paquete de torchvision:

    ```
    conda install -c pytorch torchvision
    ```

    El sistema debe informar de que todos los paquetes solicitados ya están instalados. Puede omitir las advertencias sobre la versión de Anaconda (conda).

#### Reinicio de Visual Studio y el kernel de Python

Si ha actualizado el entorno de Anaconda siguiendo los pasos anteriores o ha cerrado Visual Studio Code después de completar los ejercicios del módulo anterior, debe reiniciar la aplicación y el kernel `myenv` de Python.

1. Reinicie Visual Studio Code.
2. Abra el archivo de Jupyter Notebook que creó anteriormente.
    En los ejercicios anteriores, se usó el archivo de Jupyter Notebook *ClassifySpaceRockProgram.ipynb*.
3. Inicie el kernel `myenv` de Python de Jupyter. En las esquinas superior derecha e inferior izquierda de Visual Studio, cambie al entorno de Anaconda (`'myenv'`) que creó anteriormente.

#### Nueva ejecución de celdas en el archivo de Jupyter Notebook

Si ha actualizado el entorno de Anaconda o ha reiniciado Visual Studio Code, debe ejecutar las celdas existentes en el archivo de Jupyter Notebook para poder agregar nuevas celdas de código.

1. Para volver a ejecutar las celdas en el archivo de Jupyter Notebook, comience desde la primera celda del archivo de Notebook.
2. Ejecute cada celda de Notebook en orden, desde la primera celda del archivo hasta la última.
3. Si no hay errores, continúe con la sección siguiente, [Creación de una red neuronal]().

#### Solución de errores del entorno

Estas son algunas sugerencias para ayudar a solucionar errores en el proceso de instalación:

* Si recibe errores al ejecutar celdas existentes en el archivo de Jupyter Notebook, asegúrese de haber seguido todos los pasos de esta sección:
    - 1. Reinicie el entorno de Anaconda. Active `myenv`. Compruebe la instalación de torchvision.
    - 2. Reinicie Visual Studio Code. Inicie el kernel de Python `myenv` de Jupyter.
    - 3. Ejecute las celdas existentes en el archivo de Jupyter Notebook, desde la primera celda hasta la última.

* Si recibe un error sobre un comando o biblioteca específicos, es posible que tenga que actualizar una biblioteca a través del entorno del símbolo del sistema de Anaconda. Asegúrese de que el entorno del símbolo del sistema de Anaconda indica que todas las bibliotecas se descargan e instalan. Siga los pasos para [descargar las bibliotecas de IA de Python](https://learn.microsoft.com/es-es/training/modules/introduction-artificial-intelligence-nasa/7-install-ai-libraries?azure-portal=true) como se describe en un módulo anterior.

* Si detecta errores en Visual Studio Code, intente reiniciar la aplicación, reiniciar el kernel `myenv` y ejecutar las celdas existentes en el archivo de Jupyter Notebook.

* Si es posible, intente completar los ejercicios de todos los módulos de la ruta de aprendizaje en una sola sesión. Intente no cerrar el entorno del símbolo del sistema de Anaconda o Visual Studio Code.

### Creación de una red neuronal

Una vez que confirme que el entorno está activo, estará listo para crear una red neuronal para el modelo de inteligencia artificial.

#### Detección del tipo de dispositivo

Es necesario ayudar al equipo a determinar la manera más eficaz de crear la red de aprendizaje profundo. En primer lugar, es necesario que encontremos el tipo de dispositivo que usa: CPU o GPU. Las API de PyTorch ofrecen compatibilidad para formar una red neuronal según el tipo de dispositivo.

* Agregue el siguiente código en una nueva celda y, a continuación, ejecute la celda.

    ```python
    # Determine if you're using a CPU or a GPU device to build the deep learning network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    ```

    Una vez que el sistema detecta el dispositivo, descarga las estructuras de modelo adecuadas en la ubicación de instalación de PyTorch del equipo.

#### Creación de neuronas y conexión de la red

Agreguemos código al archivo de Jupyter Notebook para compilar las neuronas y conectar la red.

* Agregue el siguiente código en una nueva celda y, a continuación, ejecute la celda.

    ```python
    # Build all the neurons
    for param in model.parameters():
        param.requires_grad = False

    # Wire the neurons together to create the neural network
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(512, 2),
                                nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

    # Add the neural network to the device
    model.to(device)

    print('done')
    ```

    Cuando se completa la compilación, la salida del comando muestra que el proceso se ha completado:

    ```python
    done
    ```

La red neuronal avanza y retrocede muchas veces hasta que aprende las mejores asociaciones (conexiones) entre las características y los tipos de rocas.

![neural-network-training](https://learn.microsoft.com/es-es/training/modules/train-test-predictive-ai-model-nasa/media/neural-network-training.gif)

> *Crédito de visualización*: Grant Sanderson, [https://www.3blue1brown.com/](https://www.3blue1brown.com/).

<hr/>

## Ejercicio: Entrenamiento de una red neuronal para clasificar con precisión las rocas espaciales de fotos

Ahora tenemos un modelo de IA que incorpora una red neuronal. Hemos proporcionado algunos datos a nuestro programa para enseñarle las distintas características de las rocas espaciales. El programa tiene muchas neuronas y están conectadas conjuntamente en una red de aprendizaje profundo.

Ahora, es el momento de entrenar nuestro programa. Usaremos nuestros datos de entrenamiento de la NASA. Agregaremos código para ayudar a nuestro programa a ser preciso a la hora de clasificar las rocas espaciales.

### Iteración en los datos y aumento de la precisión

En esta sección de código, busque la variable `epochs`. Esta variable indica al programa cuántas veces debe buscar asociaciones en las características. En nuestro ejemplo, estableceremos el número inicial de iteraciones en 5.

Para entrenar nuestro modelo, cargamos la entrada de imagen de la variable `trainloader` que hemos creado en el módulo Análisis de imágenes de rocas mediante inteligencia artificial. Los datos se almacenan en el dispositivo ya seleccionado. Llamamos a la función `optimizer.zero_grad()` para la puesta a cero de degradados y evitar la acumulación de degradados en las iteraciones de entrenamiento.

La entrada de imagen se pasa a través del modelo mediante la función `model.forward(inputs)`, que devuelve las probabilidades de registro de cada etiqueta. La función `criterion(logps, labels)` ejecuta las probabilidades de registro a través del criterio para obtener el gráfico de salida. La función `loss.backward()` usa el gráfico de pérdida para calcular los degradados. A continuación, la función `optimizer.step()` actualiza los parámetros en función del degradado actual.

Durante el entrenamiento y las pruebas, se realiza un seguimiento de los valores de pérdida para cada iteración y el lote completo. Cada cinco `epochs`, se evalúa el modelo. Usamos la función `model.eval()` con la función `torch.no_grad()` para deshabilitar elementos del modelo con un comportamiento distinto durante el entrenamiento frente a la evaluación. Usamos este par de funciones para refinar la precisión de la predicción sin actualizar los degradados.

La función `torch.exp(logps)` se usa para obtener un nuevo tensor con las probabilidades verdaderas. La mayor probabilidad y clase del nuevo tensor a lo largo de una dimensión determinada se devuelve desde la función `ps.topk(1, dim=1)`. El tensor se cambia de forma para que coincida con la misma forma que la clase superior.

Por último, calculamos la precisión general.

### Entrenamiento de la red neuronal

Siga estos pasos para entrenar la red neuronal en el modelo de inteligencia artificial.

1. Vuelva a Visual Studio Code y abra el archivo de Jupyter Notebook. En nuestro ejemplo, se usa el archivo *ClassifySpaceRockProgram.ipynb*.

2. Asegúrese de que ejecuta el kernel de Jupyter correcto. En las esquinas superior derecha e inferior izquierda de Visual Studio, cambie al entorno de Anaconda (`'myenv'`) que creó anteriormente.

3. Agregue el siguiente código en una nueva celda y, a continuación, ejecute la celda.

    ```python
    # Set the initial number of iterations to search for associations
    epochs = 5
    print_every = 5

    # Initialize the loss variables
    running_loss = 0
    train_losses, test_losses = [], []

    # Track the current training step, start at 0
    steps = 0

    # Search for associations in the features
    for epoch in range(epochs):

        # Count each epoch
        epoch += 1

        # Load in all of the image inputs and labels from the TRAIN loader 
        for inputs, labels in trainloader:

            # Count each training step
            steps += 1
            print('Training step ', steps)

            # Load the inputs and labels to the already selected device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero out gradients to avoid accumulations of gradiants across training iterations
            optimizer.zero_grad()

            # Pass the images through the model, return the log probabilities of each label
            logps = model.forward(inputs)

            # Run the log probabilities through the criterion to get the output graph
            loss = criterion(logps, labels)

            # Use the loss graph to compute gradients
            loss.backward()

            # Update the parameters based on the current gradient
            optimizer.step()

            # Add the actual loss number to the running loss total
            running_loss += loss.item()

            # Every 5 steps, evaluate the model
            if steps % print_every == 0:

                # Initialize loss and accuracy
                test_loss = 0
                accuracy = 0

                # Start the model evaluation
                model.eval()

                # Refine the accuracy of the prediction without updating the gradients
                with torch.no_grad():

                    # Load in all of the image inputs and labels from the TEST loader 
                    for inputs, labels in testloader:

                        # Load the inputs and labels to the already selected device
                        inputs, labels = inputs.to(device), labels.to(device)

                        # Pass the images through the model, return the log probabilities of each label
                        logps = model.forward(inputs)

                        # Run the log probabilities through the criterion to get the output graph
                        batch_loss = criterion(logps, labels)

                        # Add the actual loss number to the running loss total for the test batch
                        test_loss += batch_loss.item()

                        # Return a new tensor with the true probabilities
                        ps = torch.exp(logps)

                        # Return the largest probability and class of the new tensor along a given dimension
                        top_p, top_class = ps.topk(1, dim=1)

                        # Reshape the tensor to match the same shape as the top class
                        equals = top_class == labels.view(*top_class.shape)

                        # Compute the accuracy and add it to the running accuracy count for the test batch
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Append the training and testing losses
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(testloader))  

                # Display the accuracy of the prediction with 3 digits in the fractional part of the decimal
                print(f"\n     Epoch {epoch}/{epochs}: "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}\n")

                # Train the model
                running_loss = 0
                model.train()

                # After 5 training steps, start the next epoch
                # Break here in case the trainloader has remaining data
                break
    ```

    A medida que avanza la compilación, la salida muestra cada paso de entrenamiento y época completados:

    ```
    Training step 1
    Training step 2
    Training step 3
    Training step 4
    Training step 5

        Epoch 1/5: Train loss: 0.550.. Test loss: 0.282.. Test accuracy: 0.902

    Training step 6
    Training step 7
    Training step 8
    Training step 9
    Training step 10

        Epoch 2/5: Train loss: 0.451.. Test loss: 0.311.. Test accuracy: 0.842

    Training step 11
    Training step 12
    Training step 13
    ...
    ```

    ¿Observa que la salida de cada época sucesiva tarda un poco más en mostrarse que la anterior?

### Análisis de la salida de entrenamiento

Una vez completadas cinco épocas, el sistema alcanza nuestro límite `epoch`.

```
...
Training step 19
Training step 20

     Epoch 4/5: Train loss: 0.216.. Test loss: 0.189.. Test accuracy: 0.906

Training step 21
Training step 22
Training step 23
Training step 24
Training step 25

     Epoch 5/5: Train loss: 0.234.. Test loss: 0.175.. Test accuracy: 0.935
```

La salida muestra la precisión de predicción para cada iteración de época con pérdidas de entrenamiento y prueba, y la precisión de la prueba.

Estos son los resultados de nuestra prueba con cinco épocas. Los resultados específicos serán diferentes porque el equipo elige un conjunto de imágenes aleatorias para cada ejecución de prueba. Los resultados revelan la pérdida de entrenamiento, la pérdida de prueba y la precisión. Todo ello depende de la imagen elegida.

| Época | Pérdida de entrenamiento | Pérdida de prueba | Precisión de la prueba
| ----- | ------------------------ | ----------------- | ----------------------
|   1   |           0.550          |       0.282       |          0.902
|   2   |           0.451          |       0.311       |          0.842
|   3   |           0.342          |       0.233	   |          0.902
|   4   |           0.216          |       0.189       |          0.906
|   5   |           0.234          |       0.175	   |          0.935

<hr/>

## Ejercicio: Determinar la precisión de una red neuronal en la clasificación de rocas espaciales

La razón por la que usamos la inteligencia artificial para ayudar con nuestra investigación de rocas espaciales es la mejora de las predicciones. Cuando un modelo ve nuevas imágenes de rocas, debe predecir el tipo de roca correcto en función de las imágenes que ya ha visto. Queremos que la predicción sea lo más precisa posible. Un modelo de inteligencia artificial bien entrenado debe predecir con más precisión el resultado que las personas.

### Mostrar la precisión del modelo

En nuestro ejemplo, el comando `accuracy` revela la probabilidad de que el equipo pueda identificar correctamente el tipo de roca en una imagen en función de la definición científica. Un valor de precisión de 0,96 significa que el 96% de los tipos de roca se predicen correctamente y que el 4% se clasifican incorrectamente.

El código siguiente calcula y muestra la precisión de nuestro modelo de IA para clasificar el tipo de roca. Es posible que reconozca esta instrucción del último código que agregamos al archivo de Jupyter Notebook.

* Agregue este código en una nueva celda del archivo de Jupyter Notebook y, a continuación, ejecute la celda:

    ```python
    print(accuracy/len(testloader))
    ```

La salida muestra la precisión con 16 dígitos en la parte fraccionaria del decimal:

    ```python
    0.9354166686534882
    ```

La salida muestra que nuestro modelo tiene una precisión del 93,5% al realizar predicciones. Queremos un valor alto. Cuanto mayor sea la precisión, mejor será el trabajo del modelo realizando predicciones.

Aunque el 93,5% es alto, puede hacer algunas cosas para aumentar aún más la precisión:

* Agregue más imágenes y continúe entrenando el modelo de IA.
* Aumente el número de iteraciones de entrenamiento `epoch` para aprendizaje profundo.

### Guardar el modelo

Ahora que hemos creado la red neuronal y probado la precisión, guarde el modelo.

* Agregue el código siguiente en una nueva celda del archivo de Jupyter Notebook y, a continuación, ejecute la celda:

    ```python
    torch.save(model, 'aerialmodel.pth')
    ```

## Ejercicio: Predicción del tipo de roca espacial en una fotografía aleatoria

Usemos nuestro modelo para predecir los tipos de roca.

Para predecir el tipo de roca en una imagen nueva es necesario completar estos pasos:

* Paso 1: Convertir la imagen nueva en números. Use la función `test_transforms` que creó para transformar las imágenes en el modelo.
* Paso 2: Transformar la imagen. Recorte y cambie el tamaño de la imagen a 224 x 224 píxeles con las funciones `unsqueeze` y `Variable`.
* Paso 3: Extraer las características de la imagen. Pase la imagen al modelo para realizar las extracciones.
* Paso 4: Predecir el tipo de roca que se muestra en la imagen. Use las asociaciones que aprendimos en el paso 2 mediante la búsqueda de la predicción de probabilidad más alta a partir de los resultados del modelo.

### Uso del modelo para realizar predicciones

Siga estos pasos para realizar predicciones con la red neuronal en el modelo de inteligencia artificial.

1. En primer lugar, cargamos la red neuronal. Agregue el código siguiente en una nueva celda del archivo de Jupyter Notebook y, a continuación, ejecute la celda:

    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.load('aerialmodel.pth')
    ```

2. A continuación, creamos una función para predecir el tipo de roca en una nueva imagen comparándolo con el patrón de matriz de nuestro modelo. Agregue el código siguiente en una nueva celda del archivo de Jupyter Notebook y, a continuación, ejecute la celda:

    ```python
    def predict_image(image):
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(device)
        output = model(input)
        index = output.data.cpu().numpy().argmax()
        return index
    ```

Ha definido la función de predicción de imágenes. Ahora puede continuar con el ejercicio final y llamar a esta función para predecir el tipo de roca de una imagen.

<hr/>

## Ejercicio: probar una red neuronal que clasifique fotos de rocas espaciales

Por último, está listo para probar el modelo de IA completo. Veamos lo bien que el modelo puede predecir el tipo de roca correcto.

Para empezar, indicaremos al equipo que seleccione cinco imágenes de forma aleatoria para la primera prueba. Puede elegir cualquier número de imágenes para probar. Después de la primera prueba, volveremos a ejecutar el código con más imágenes.

Usamos los datos de la variable `trainloader` una vez más y pasamos cada imagen a nuestras funciones para extraer características de la roca de la foto. El equipo compara las características con los patrones reconocidos por el modelo. Con esa información, el modelo predice el tipo de roca de la foto. El último paso es usar la función `plt` para trazar un gráfico de los resultados de nuestra predicción.

### Predicción de los tipos de roca de imágenes aleatorias

Siga estos pasos para probar la precisión de predicción de la red neuronal en el modelo de IA.

1. Agregue el código siguiente en una nueva celda del archivo de Jupyter Notebook y, a continuación, ejecute la celda:

    ```python
    # Get five random images and display them in a figure with their labels
    to_pil = transforms.ToPILImage()
    images, labels = get_random_images(5)
    fig=plt.figure(figsize=(20,10))

    # Load all of the classes from the training loader
    classes=trainloader.dataset.classes

    # Loop through the 5 randomly selected images
    for ii in range(len(images)):

        # Predict the class of each image
        image = to_pil(images[ii])
        index = predict_image(image)

        # Add the class to the plot graph to display beneath the image
        sub = fig.add_subplot(1, len(images), ii+1)
        res = int(labels[ii]) == index
        sub.set_title(str(classes[index]) + ":" + str(res))
        plt.axis('off')
        plt.imshow(image)

    # Reshow the plot with the predicted labels beneath the images
    plt.show()
    ```

    Este código crea un objeto visual de las imágenes con etiquetas para mostrar el tipo de roca real y la predicción del modelo: True o False. La predicción muestra si el sistema de inteligencia artificial ha clasificado correctamente el tipo de roca.

    ![test-model-prediction-accuracy](https://learn.microsoft.com/es-es/training/modules/train-test-predictive-ai-model-nasa/media/test-model-prediction-accuracy.png)

2. Pruebe otra prueba. En la celda que agregó en el paso anterior, cambie el número de imágenes que se van a probar a **10** y, a continuación, vuelva a ejecutar la celda:

    ```python
    ...
    images, labels = get_random_images(10)
    ...
    ```

    ¿Observa alguna mejora en la precisión?

3. Presione `Ctrl + S` para guardar los cambios en el archivo de Jupyter Notebook.

¡Enhorabuena! Ha creado correctamente una red neuronal en funcionamiento para predecir el tipo de un objeto de una imagen. Tiene un modelo de inteligencia artificial que clasifica los tipos de rocas lunares recopiladas por la NASA.

La inteligencia artificial combina grandes cantidades de datos con formas creativas de comprender, clasificar y contextualizar. Los científicos usan la inteligencia artificial como ayuda para mejorar su análisis y llegar a conclusiones correctas. Si tiene la oportunidad de convertirse en un experto en rocas espaciales, puede aprender finalmente a clasificar imágenes de rocas. Cuando usa la inteligencia artificial en la investigación, se beneficia de la experiencia de científicos que ya han estado en la Luna y han vuelto.