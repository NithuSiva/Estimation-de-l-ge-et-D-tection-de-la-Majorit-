import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tensorflow as tf
import random
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.preprocessing import image



def charger_image(image_path, img_size):
    img = load_img(image_path, target_size=(img_size, img_size))  
    img_array = img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 
    return img_array

def info(train_dir, test_dir):
    print('')
    print("* -----------INFO-------------- *")
    print("Python 3.9.0 / prediction.py")
    image_count = len(list(train_dir.glob('*/*.jpg')))
    print('')
    print("Nombre d'images dans le data/train: ", image_count)
    print('')
    image_count = len(list(test_dir.glob('*/*.jpg')))
    print('')
    print("Nombre d'images dans le data/test: ", image_count)
    print('')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("* -----------FIN INFO-------------- *")
    print('')

def graphFolder(folder_path):
    image_counts = {}

    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            image_counts[subdir] = len(os.listdir(subdir_path))

    sorted_image_counts = dict(sorted(image_counts.items(), key=lambda item: item[1], reverse=True))

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_image_counts.keys(), sorted_image_counts.values(), color='skyblue')
    plt.xlabel('Dossiers')
    plt.ylabel('Nombre d\'images')
    plt.title('Nombre d\'images par dossier')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def prediction(model_path, folder, img_size, class_names, predict_majeur_mineur):

   

    model = load_model(model_path)

    image_files = random.sample(os.listdir(folder), 9)

    images = []
    titles = []
    for image_file in image_files:
        image_path = os.path.join(folder, image_file)
        img = image.load_img(image_path, target_size=(img_size, img_size))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0) 

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_class_name = class_names[predicted_class[0]]

        images.append(img)
        if(predict_majeur_mineur):
            if(predicted_class_name in  ['0_10', '10_18']):
                titles.append(f"{image_file}\nPredicted: Mineur")
            else:
                titles.append(f"{image_file}\nPredicted: Majeur")

        else:
            titles.append(f"{image_file}\nPredicted: {predicted_class_name}")

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():


    train_path = "dataset/age_prediction_up/age_prediction/train"
    test_path = "dataset/age_prediction_up/age_prediction/test"
    train_dir = pathlib.Path(train_path)
    test_dir = pathlib.Path(test_path)
    info(train_dir, test_dir)

    # image_folder = 'ethnie_1'  
    # image_folder = 'ethnie_2'  
    # image_folder = 'homme'  
    # image_folder = 'femme'  
    image_folder = 'mixte'  

    # Modèle sur entraîné classe 0_10
    # models_path = 'models/1'  
    # model_filename = 'model_normal_01_06_2.h5' 
    # img_size = 448

    # Modèle déséquilibré  
    # models_path = 'models'  
    # model_filename = 'model_normal_28_05_2.h5' 
    # img_size = 224

     # Modèle équilibré 51%
    # models_path = 'models'  
    # model_filename = 'model_normal_29_05_2.h5' 
    # img_size = 224

    # Modèle équilibré 59%
    models_path = 'models'  
    model_filename = 'model_normal_30_05_2.h5' 
    img_size = 224

    model_path= os.path.join(models_path, model_filename)

    class_names = ['0_10', '10_18', '18_30', '30_50', '50_70', '70_100']

    prediction(model_path, image_folder, img_size, class_names, False)

    # graphFolder(test_folder_path)
        
main()