from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Input, GlobalAveragePOoling2D
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from keras.applications.vgg16.inception_v3 import InceptionV3
from keras import backend as K


batch_size = 16
N_CLASSES = 200

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate

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


def f1(y_true, y_pred):
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
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    print(precision, recall)
    return (2*((precision*recall)/(precision+recall+K.epsilon())))


# base_model

base_model = InceptionV3(weights='imagenet', include_top=False)

'''
include_top=False removes the top3 layers of InceptionV3
'''

x = base_model.output
x = GlobalAveragePOoling2D()(x)
x = Dense(1024, activation='relu')(x)

predictions= Dense(N_CLASSES, activation='softmax')(x)

Model = model(inputs=base_model.input, outputs=predictions)

for layers in model.layers[:150]:
	layer.trainable=False

for layers in model.layers[150:]:
	layer.trainable=True


from keras.optimizers import SGD

sgd = SGD(lr=0.00001, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[precision, recall, f1])


train_datagen = ImageDataGenerator(rescale=1./255)

valid_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'train/',  # this is the target directory
        target_size=(416, 416),  # images resized to 1000*1000
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
        'test/',  
        target_size=(416, 416),  # images  resized to 1000*1000
        batch_size=batch_size,
        class_mode='categorical')

filepath = "wts/vgg/vgg16-1-{epoch:02d}-{val_f1:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose =1, save_best_only=True, mode='max', save_weights_only=True)
early_stopper = EarlyStopping(monitor='val_f1', verbose=1, patience=3)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
lrate = LearningRateScheduler(step_decay)
callback_list = [checkpoint, early_stopper, lrate, tensorboard]

history = training.fit_generator(
        train_generator,
        steps_per_epoch=1600//batch_size,
        epochs=20,
        validation_data=validation_generator,
        # validation_steps=800 // batch_size
        callbacks= callback_list
        )