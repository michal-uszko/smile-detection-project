from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

BATCH_SIZE = 128
TARGET_SIZE = (224, 224)
EPOCHS = 20
DATASET_PATH = "data/dataset/"

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=10,
                                   zoom_range=0.05,
                                   shear_range=0.15,
                                   fill_mode="nearest",
                                   horizontal_flip=False,)


training_set = train_datagen.flow_from_directory(DATASET_PATH + 'train',
                                                 target_size=TARGET_SIZE,
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='binary',
                                                 color_mode='rgb')

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(DATASET_PATH + 'test',
                                            target_size=TARGET_SIZE,
                                            batch_size=BATCH_SIZE,
                                            class_mode='binary',
                                            color_mode='rgb')

base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

head_model = base_model.output
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(units=128, activation="relu")(head_model)
head_model = Dense(units=64, activation="relu")(head_model)
head_model = Dense(units=1, activation="sigmoid")(head_model)

model = Model(inputs=base_model.inputs, outputs=head_model)


for layer in base_model.layers:
    layer.trainable = False


opt = Adam(learning_rate=0.001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(x=training_set, validation_data=test_set, epochs=EPOCHS, callbacks=[early_stop])

test_set.reset()
training_set.reset()


for layer in base_model.layers[48:]:
    layer.trainable = True


opt = Adam(learning_rate=0.0001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(x=training_set, validation_data=test_set, epochs=EPOCHS, callbacks=[early_stop])

model.save('model/classifier.h5')
labels = training_set.class_indices
print(f"Labels: {labels}")
