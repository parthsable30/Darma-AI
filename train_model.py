import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Dataset Path
dataset_path = "dataset"

# Image Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Data Augmentation & Validation Split
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% data for validation
)

# Load Training Data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load Validation Data
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load Pretrained MobileNetV2 Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(len(train_generator.class_indices), activation='softmax')(x)  # Output Layer

# Create Model
model = Model(inputs=base_model.input, outputs=x)

# Freeze Base Model Layers Initially
for layer in base_model.layers:
    layer.trainable = False

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Initial Model (Feature Extraction)
model.fit(train_generator, validation_data=val_generator, epochs=5)

# Unfreeze Deeper Layers for Fine-Tuning
for layer in base_model.layers[-20:]:  # Unfreezing last 20 layers
    layer.trainable = True

# Compile Again with Lower Learning Rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue Training (Fine-Tuning)
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save Model
model.save("skin_disease_model.h5")

print("✅ Model training complete! File saved as skin_disease_model.h5")
