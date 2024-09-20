# Objectif

Ce projet innovant a pour but de faire un algorithmes permettant une estimation de l’age d’un individu sur une photo à l’aide de l’Intelligence Artificielle.

# Dataset

Dans un premier temps, après avoir récupérée notre dataset depuis kaggle sous format zip, nous remarquons qu’il est organisé par âge, c’est-à-dire qu' un dossier correspond à un âge allant de 1 ans à 100 ans.

Nous avons constaté un déséquilibre dans la répartition des classes, certaines contenant significativement plus d'images que d'autres.

![nombre_image_dossier](https://github.com/user-attachments/assets/26878dee-2655-408a-9d4a-5ae039c4967b)

Avoir plus de 100 classes s'est avéré problématique, car cela complique l'entraînement de notre modèle,  Dans notre cas, l'objectif étant de prédire si un individu est majeur, ainsi seulement 2 classes suffisent mais avoir 6 classes permet de mieux distinguer les âges proches de 18 ans, réduisant ainsi les erreurs de classification aux limites de la majorité. Nous avons par la suite réorganisé les données afin de réduire le nombre de classes, les divisant ainsi en six catégories distinctes.


![dataset_trier](https://github.com/user-attachments/assets/c345a584-64b9-44de-83fb-8eedc5f1bfe1)

# Entraînement du modèle

Nous avons utilisé un réseau de neurones convolutionnels (CNN), le CNN est permet de faire de la classification d’images, en effet il est un algorithme puissant permettant de faire du deep learning. 

## Architecture du CNN
```python
model = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes)
])
```

## Précision

Nous avons varié les hyperparamètres, tels que le batch size, la rotation de l’image, la taille de l’image et le nombre de filtres, et réalisé plusieurs tentatives d’entraînements d'une durée variant de 5 à 8 heures. Le meilleur résultat obtenu est une précision de 0.59 en entraînement et en validation. Nous avons effectué ces entraînements sur notre machine locale, car l'utilisation du TPU de Google Colab est limitée à 4 heures.

## Résultat 
Nous avons obtenu 6 prédictions correctes sur 9, et avec une dizaine d'essais, nous atteignons environ 5,2 prédictions correctes sur 9, ce qui correspond à la précision de notre modèle.
![prediction_2](https://github.com/user-attachments/assets/fd5af7e7-9b3e-4369-b3a4-19339ed5e5e5)

