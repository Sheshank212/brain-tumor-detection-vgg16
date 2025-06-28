# Enhanced VGG16-based Brain Tumor Detection Script (with Cross-Validation, Checkpointing, Fine-Tuning)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, AveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

from imutils import paths

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

plt.style.use("ggplot")
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

def load_dataset(dataset_dir, image_size=(224, 224)):
    images, labels = [], []
    for image_path in paths.list_images(dataset_dir):
        label = os.path.basename(os.path.dirname(image_path))
        img = cv2.imread(image_path)
        img = cv2.resize(img, image_size)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

path = "/Users/sheshank/Desktop/ml projects/brain tumor detection/brain_tumor_dataset"
X, y = load_dataset(path)
X = preprocess_input(X.astype("float32"))

encoder = LabelBinarizer()
y_encoded = encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
accuracies = []
fold = 1

for train_idx, test_idx in skf.split(X, np.argmax(y_encoded, axis=1)):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
    class_weight_dict = dict(enumerate(class_weights))

    augmentor = ImageDataGenerator(rotation_range=20, zoom_range=0.1, horizontal_flip=True)

    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    for layer in base_model.layers[:-8]:
        layer.trainable = False

    head = base_model.output
    head = AveragePooling2D(pool_size=(4, 4))(head)
    head = Flatten()(head)
    head = Dense(64, activation="relu")(head)
    head = BatchNormalization()(head)
    head = Dropout(0.4)(head)
    head = Dense(2, activation="softmax")(head)

    model = Model(inputs=base_model.input, outputs=head)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    checkpoint = ModelCheckpoint(
        filepath=f"models/best_model_fold{fold}.keras",
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6)

    history = model.fit(
        augmentor.flow(X_train, y_train, batch_size=8),
        validation_data=(X_test, y_test),
        epochs=25,
        callbacks=[early_stop, reduce_lr, checkpoint],
        class_weight=class_weight_dict
    )
    
    # Plot training history for this fold
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Fold {fold} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"Fold {fold} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"results/training_plot_fold{fold}.jpg")
    plt.show()
    plt.close()

    model = load_model(f"models/best_model_fold{fold}.keras")

    predictions = model.predict(X_test, batch_size=8)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    acc = np.trace(conf_matrix) / float(np.sum(conf_matrix))
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    accuracies.append(acc)

    report = classification_report(true_classes, predicted_classes, target_names=encoder.classes_)
    print(report)
    with open(f"results/classification_report_fold{fold}.txt", "w") as f:
        f.write(report)

    fold += 1

print("\nAverage Cross-Validated Accuracy:", np.mean(accuracies))

# Load best model for final evaluation and visualization
best_model = load_model("models/best_model_fold1.keras")

# Use the last fold's test data for visualization
X_final_test, y_final_test = X[test_idx], y_encoded[test_idx]

# Final comprehensive evaluation
final_predictions = best_model.predict(X_final_test, batch_size=8)
final_predicted_classes = np.argmax(final_predictions, axis=1)
final_true_classes = np.argmax(y_final_test, axis=1)

# Final classification report
final_report = classification_report(final_true_classes, final_predicted_classes, target_names=encoder.classes_)
print("\n=== Final Model Evaluation ===")
print(final_report)
with open("results/final_classification_report.txt", "w") as f:
    f.write(final_report)

# Final confusion matrix
final_conf_matrix = confusion_matrix(final_true_classes, final_predicted_classes)
final_accuracy = np.trace(final_conf_matrix) / float(np.sum(final_conf_matrix))
print(f"Final Model Accuracy: {final_accuracy:.4f}")

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(final_conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Final Model Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix_heatmap.jpg")
plt.show()
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(final_true_classes, final_predictions[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/roc_curve.jpg")
plt.show()
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(final_true_classes, final_predictions[:, 1])
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("results/precision_recall_curve.jpg")
plt.show()
plt.close()

# Model Architecture Visualization
plot_model(best_model, to_file="results/vgg16_brain_model.png", show_shapes=True, show_layer_names=True)

# Grad-CAM Implementation
def generate_gradcam(img_path, model, class_idx, save_path="results/grad_cam.jpg"):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    grad_model = Model(inputs=model.input, outputs=[model.get_layer("block5_conv3").output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs, weights.numpy())

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Load and preprocess original image for overlay
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))
    superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    result_img = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    plt.imshow(result_img)
    plt.axis("off")
    plt.title("Grad-CAM Visualization")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

# Example prediction with Grad-CAM
predict_image = "/Users/sheshank/Desktop/ml projects/brain tumor detection/brain_tumor_dataset/yes/Y1.jpg"
test_img = cv2.imread(predict_image)
test_img = cv2.resize(test_img, (224, 224))
test_img_processed = preprocess_input(np.expand_dims(test_img.astype("float32"), axis=0))

pred = best_model.predict(test_img_processed)
predicted_class = encoder.classes_[np.argmax(pred)]
confidence = np.max(pred)

print(f"\n=== Example Prediction ===")
print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")

# Generate Grad-CAM for the predicted class
generate_gradcam(predict_image, best_model, class_idx=np.argmax(pred))
