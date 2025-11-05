import os
from keras.models import Model, load_model
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from glob import glob

# ✅ Use your actual Windows paths (raw strings recommended)
training_data = r"C:\chest_xray\chest_xray\train"
testing_data  = r"C:\chest_xray\chest_xray\test"
validation_data = r"C:\chest_xray\chest_xray\val"

IMAGESHAPE = [224, 224, 3]

# ✅ Load pretrained VGG16 model
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)

for each_layer in vgg_model.layers:
    each_layer.trainable = False  # Freeze VGG16 layers

# ✅ Dynamically detect class count
classes = glob(training_data + "/*")

# Add custom layers
flatten_layer = Flatten()(vgg_model.output)
prediction = Dense(len(classes), activation='softmax')(flatten_layer)
final_model = Model(inputs=vgg_model.input, outputs=prediction)

final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ✅ Data augmentation setup
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# ✅ Flow data from directories
training_set = train_datagen.flow_from_directory(
    training_data,
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical'
)

validation_set = val_datagen.flow_from_directory(
    validation_data,
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    testing_data,
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical'
)

# ✅ Train the model
fitted_model = final_model.fit(
    training_set,
    validation_data=validation_set,
    epochs=5,
    steps_per_epoch=training_set.samples // training_set.batch_size,
    validation_steps=validation_set.samples // validation_set.batch_size
)

# ✅ Save the trained model
final_model.save('pneumonia_model.keras')
