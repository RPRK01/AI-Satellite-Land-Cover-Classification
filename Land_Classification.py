import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ---------------- CONFIG ----------------
IMG_SIZE = 160
BATCH_SIZE = 8
EPOCHS = 5
SAMPLES_PER_CLASS = 3
DATASET_PATH = "dataset"

COLORS = {
    "urban": "red",
    "vegetation": "green",
    "water": "blue"
}

# ---------------- LOAD DATA ----------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ---------------- MODEL ----------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- TRAIN ----------------
model.fit(train_ds, epochs=EPOCHS)

# ---------------- COLLECT PREDICTIONS ----------------
images_list = []
preds_list = []

for images, labels in train_ds:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)
    images_list.extend(images.numpy())
    preds_list.extend(preds)

images_list = np.array(images_list)
preds_list = np.array(preds_list)

# ---------------- DISPLAY 3x3 COLLAGE ----------------
plt.figure(figsize=(9, 9))
plot_id = 1

for class_idx, class_name in enumerate(class_names):
    idxs = np.where(preds_list == class_idx)[0][:SAMPLES_PER_CLASS]

    for idx in idxs:
        plt.subplot(3, 3, plot_id)
        plt.imshow(images_list[idx].astype("uint8"))
        plt.title(class_name.upper(), color=COLORS[class_name])
        plt.axis("off")
        plot_id += 1

plt.suptitle(
    "AI-Based Satellite Image Classification\nUrban / Vegetation / Water",
    fontsize=14
)
plt.tight_layout()
plt.show()

