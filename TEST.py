import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import logging
import argparse

def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def predict_image(model, path, size, breed2id, id2breed):
    image = read_image(path, size)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    label_idx = np.argmax(pred)
    return id2breed[label_idx]

def main(base_path, model_path, save_dir, num_images_to_process):
    # Set up logging
    logging.basicConfig(filename='testing_log.log', level=logging.DEBUG, 
                        format='%(asctime)s:%(levelname)s:%(message)s')

    train_path = os.path.join(base_path, "train\\*")
    test_path = os.path.join(base_path, "test\\*")
    labels_path = os.path.join(base_path, "labels.csv")

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    print("Number of Breed: ", len(breed))
    logging.debug(f"Number of Breed: {len(breed)}")

    breed2id = {name: i for i, name in enumerate(breed)}
    id2breed = {i: name for i, name in enumerate(breed)}

    ids = glob(train_path)
    labels = []

    for image_path in ids:
        image_id = os.path.basename(image_path).split(".")[0]
        breed_name_list = list(labels_df[labels_df.id == image_id]["breed"])
        if breed_name_list:
            breed_name = breed_name_list[0]
            breed_idx = breed2id[breed_name]
            labels.append(breed_idx)
        else:
            logging.warning(f"No breed found for image_id: {image_id}")

    ids = ids[:5000]
    labels = labels[:5000]
    
    # Splitting the dataset
    if len(ids) < 5:
        raise ValueError("Not enough samples to split the dataset. Add more data or adjust the split parameters.")
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(ids, labels, test_size=0.2, random_state=42)

    # Load the trained model from the specified model path
    model = tf.keras.models.load_model(model_path)
    logging.debug(f"Model loaded from: {model_path}")

    save_dir = os.path.join(base_path, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    logging.debug(f"Save directory created or already exists: {save_dir}")

    true_labels = []
    predictions = []

    batch_size = 32
    for start in tqdm(range(0, len(valid_x[:num_images_to_process]), batch_size)):
        end = min(start + batch_size, len(valid_x[:num_images_to_process]))
        batch_paths = valid_x[start:end]
        batch_images = [read_image(path, 224) for path in batch_paths]
        batch_images = np.array(batch_images)
        batch_preds = model.predict(batch_images)

        for i, path in enumerate(batch_paths):
            try:
                pred = batch_preds[i]
                label_idx = np.argmax(pred)
                breed_name = id2breed[label_idx]

                ori_breed = id2breed[valid_y[start + i]]
                true_labels.append(valid_y[start + i])
                predictions.append(label_idx)
                
                ori_image = cv2.imread(path, cv2.IMREAD_COLOR)
                ori_image = cv2.putText(ori_image, breed_name, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                ori_image = cv2.putText(ori_image, ori_breed, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                save_path = os.path.join(save_dir, f"valid_{start + i}.png")
                success = cv2.imwrite(save_path, ori_image)
                
                if success:
                    logging.debug(f"Image saved successfully: {save_path}")
                else:
                    logging.error(f"Failed to save image: {save_path}")
            except Exception as e:
                logging.error(f"Error processing image {path}: {e}")

    # Calculate and print the overall accuracy
    accuracy = np.mean(np.array(true_labels) == np.array(predictions))
    logging.info(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Generate and print the classification report
    report = classification_report(true_labels, predictions, target_names=breed, zero_division=0)
    logging.info(f"Classification Report:\n{report}")
    print(report)

    # Generate and plot the confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    logging.info(f"Confusion Matrix:\n{cm}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    # Check if the script is running in an interactive environment
    try:
        get_ipython()
        interactive_env = True
    except NameError:
        interactive_env = False

    if interactive_env:
        # Set default values for testing in an interactive environment
        base_path = r"C:\Users\User\Documents\Murdoch University_Sandeep 2023\3rd Semester\Artificial Intelligence\Assignment 2\Dog-Breed-Classification"
        model_path = os.path.join(base_path, "models", "final_model.keras")
        save_dir = "save"
        num_images_to_process = 1000
    else:
        # Use argparse for command-line arguments
        parser = argparse.ArgumentParser(description="Dog Breed Classification Testing Script")
        parser.add_argument("--base_path", type=str, required=True, help="Base path to the dataset")
        parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
        parser.add_argument("--save_dir", type=str, default="save", help="Directory to save the result images")
        parser.add_argument("--num_images_to_process", type=int, default=1000, help="Number of images to process")
        args = parser.parse_args()

        base_path = args.base_path 
        model_path = args.model_path
        save_dir = args.save_dir
        num_images_to_process = args.num_images_to_process

    main(base_path, model_path, save_dir, num_images_to_process)
