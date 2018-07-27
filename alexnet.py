from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Input, BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from time import time
import math
from keras import backend as K


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    print(precision, recall)
    return (2*((precision*recall)/(precision+recall+K.epsilon())))

batch_size = 16
# input_size = (416, 416, 3)
# nb_classes = 16
# mean_flag = False

def alex():
    v1 = Input(shape=(416, 416, 3))

    # 1st Layer
    conv_1 = Conv2D(96, (11, 11), strides=(4,4), padding='valid')(v1)
    conv_1 = Activation('relu')(conv_1)
    conv_1 = BatchNormalization()(conv_1)  
    conv_1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid')(conv_1)
 
    # 2nd Layer
    conv_2 = Conv2D(256, (5, 5))(conv_1)
    conv_2 = Activation('relu')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(conv_2)

    # 3rd Layer
    conv_3 = Conv2D(384, (3, 3))(conv_2)
    # conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)
    # conv_3 = MaxPooling2D((3, 3))(conv_3)

    # 4th Layer
    conv_4 = Conv2D(384, (3, 3))(conv_3)
    # conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)
    # conv_4 = MaxPooling2D((3, 3))(conv_4)

    # 5th Layer
    conv_5 = Conv2D(256, (3, 3))(conv_4)
    conv_5 = Activation('relu')(conv_5)
    conv_5 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(conv_5)
    
    # Flatten Layer
    f = Flatten()(conv_5)

    # 6th Layer
    dense_1 = Dense(4096)(f)
    # dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu')(dense_1)
    drop_1  = Dropout(0.4)(dense_1)

    # 7th Layer
    dense_2 = Dense(4096)(drop_1)
    # dense_2 = BatchNormalization()(dense_2)
    dense_2 = Activation('relu')(dense_2)
    drop_2  = Dropout(0.4)(dense_2)

    dense_3 = Dense(16)(drop_2)

    predictions = Activation('softmax')(dense_3)

    model = Model(inputs=v1, outputs=predictions)
    
    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[f1, 'accuracy'])

    return model


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(416, 416),  # images resized to 1000*1000
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'data/test',  
        target_size=(416, 416),  # images  resized to 1000*1000
        batch_size=batch_size,
        class_mode='categorical')


# Saving Best Models
filepath = "alex/alex-2-{epoch:02d}-{val_f1:.2f}.hdf5"

# Callbacks
checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose =1, save_best_only=True, mode='max', save_weights_only=True)
early_stopper = EarlyStopping(monitor='val_acc', verbose=1, patience=3)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
lrate = LearningRateScheduler(step_decay)
callback_list = [checkpoint, early_stopper, tensorboard]

# Model
training = alex()

# Training
training.fit_generator(
        train_generator,
        steps_per_epoch=1600 // batch_size,
        epochs=15,
        validation_data=validation_generator,
        # validation_steps=800 // batch_size
        callbacks= callback_list
        )

# Save Model weights
training.save_weights('alex.h5')