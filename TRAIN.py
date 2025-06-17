import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

def build_model(size, num_classes):
    inputs = Input((size, size, 3))
    backbone = ResNet50(input_tensor=inputs, include_top=False, weights="imagenet")
    backbone.trainable = True
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, x)
    return model

def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def parse_data(x, y):
    x = x.decode()
    num_class = 120
    size = 224
    image = read_image(x, size)
    label = [0] * num_class
    label[y] = 1
    label = np.array(label)
    label = label.astype(np.int32)
    return image, label

def tf_parse(x, y):
    x, y = tf.numpy_function(parse_data, [x, y], [tf.float32, tf.int32])
    x.set_shape((224, 224, 3))
    y.set_shape((120,))
    return x, y

def data_augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.1)
    zoom_factor = tf.random.uniform([], 0.8, 1.2)
    new_size = tf.cast(tf.cast(tf.shape(image)[:2], tf.float32) * zoom_factor, tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    return image

def tf_dataset(x, y, batch=8, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    if augment:
        dataset = dataset.map(lambda x, y: (data_augment(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset

if __name__ == "__main__":
    base_path = r"C:\Users\User\Documents\Murdoch University_Sandeep 2023\3rd Semester\Artificial Intelligence\Assignment 2\Dog-Breed-Classification"
    train_path = os.path.join(base_path, "train\\*")
    test_path = os.path.join(base_path, "test\\*")
    labels_path = os.path.join(base_path, "labels.csv")

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    print("Number of Breed: ", len(breed))

    breed2id = {name: i for i, name in enumerate(breed)}

    ids = glob(train_path)
    labels = []

    for image_path in ids:
        image_id = os.path.basename(image_path).split(".")[0]
        print("Current image ID:", image_id)
        breed_name_list = list(labels_df[labels_df.id == image_id]["breed"])
        if breed_name_list:
            breed_name = breed_name_list[0]
            breed_idx = breed2id[breed_name]
            labels.append(breed_idx)
        else:
            print(f"No breed found for image_id: {image_id}")

    ids = ids[:10000]
    labels = labels[:10000]
    
    ## Splitting the dataset
    if len(ids) < 5:
        raise ValueError("Not enough samples to split the dataset. Add more data or adjust the split parameters.")
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(ids, labels, test_size=0.2, random_state=42)

        ## Parameters
        size = 224
        num_classes = 120
        lr = 1e-4
        batch = 16
        epochs = 10

        ## Model
        model = build_model(size, num_classes)
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr), metrics=["acc"])

        ## Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
        class_weights = {i: class_weights[i] for i in range(len(class_weights))}

        ## Dataset
        train_dataset = tf_dataset(train_x, train_y, batch=batch, augment=True)
        valid_dataset = tf_dataset(valid_x, valid_y, batch=batch, augment=False)

        ## Training
        callbacks = [
            ModelCheckpoint("model_checkpoint.keras", verbose=1, save_best_only=True, save_weights_only=False),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
            EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        ]
        train_steps = (len(train_x)//batch) + 1
        valid_steps = (len(valid_x)//batch) + 1
        history = model.fit(train_dataset,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            validation_data=valid_dataset,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks)

        # Create a directory to save the model if it does not exist
        model_save_dir = os.path.join(base_path, "models")
        os.makedirs(model_save_dir, exist_ok=True)

        # Save the final model in the specified directory
        model_save_path = os.path.join(model_save_dir, "final_model.keras")
        model.save(model_save_path)

        # Verify that the file exists
        assert os.path.exists(model_save_path), "Model file was not saved correctly."

        # Plotting the learning curve
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['acc'], label='Training Accuracy')
        plt.plot(history.history['val_acc'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.show()
