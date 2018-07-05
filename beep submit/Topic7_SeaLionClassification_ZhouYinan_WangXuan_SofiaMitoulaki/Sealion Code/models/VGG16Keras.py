from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import keras
import sklearn
from keras.optimizers import Adam
import os


class VGG16Keras():

    def __init__(self):

        model = applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(96, 96, 3))

        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dense(6, activation='softmax'))

        self.model = keras.Model(inputs=model.input, outputs=top_model(model.output))


if __name__ == '__main__':

    # some setting
    train = True
    evaluate = False
    load_from_past = False
    path = os.getcwd() + '/'

    # Hyper parameters
    learning_rate = 1e-4
    batch_size = 60
    n_epoch = 10

    # model
    net = VGG16Keras()
    model = net.model
    if load_from_past:
        model_path = path + 'saved/VGG16Keras_best.h5'
        try:
            model = keras.models.load_model(model_path)
            print('previous model loaded successfully!')
        except:
            print('previous model loading failed, start a new model!')

    if train:
        print('training...')
        # get data
        trainset = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode="nearest",
            horizontal_flip=True,
            rescale=1. / 255
        )

        valset = ImageDataGenerator(
            rescale=1. / 255)

        train_gen = trainset.flow_from_directory(
            path + 'data/Patch/train/',
            target_size=(96, 96),
            batch_size=batch_size
        )

        val_gen = valset.flow_from_directory(
            path + 'data/Patch/valid/',
            target_size=(96, 96),
            batch_size=batch_size
        )

        weights = sklearn.utils.compute_class_weight(class_weight='balanced',
                                                     classes=[0, 1, 2, 3, 4, 5],
                                                     y=train_gen.classes)
        class_weights = {}
        for x in [0, 1, 2, 3, 4, 5]:
            class_weights[x] = weights[x]

        # checkpoint
        checkpoint_each = keras.callbacks.ModelCheckpoint(path + 'saved/SealionClassifyKeras_each.h5',
                                                          verbose=1, save_best_only=False, mode='auto')

        checkpoint_best = keras.callbacks.ModelCheckpoint(path + 'saved/SealionClassifyKeras_best.h5',
                                                          verbose=1, save_best_only=True, mode='auto')

        # complie
        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])

        # train
        model.fit_generator(
            train_gen,
            steps_per_epoch=train_gen.n // batch_size,
            epochs=n_epoch,
            validation_data=val_gen,
            validation_steps=val_gen.n // batch_size,
            class_weight=class_weights,
            callbacks=[checkpoint_best, checkpoint_each]
        )

    if evaluate:
        print('evaluating...')

        testset = ImageDataGenerator(
            rescale=1. / 255)

        test_gen = testset.flow_from_directory(
            path + 'data/Patch/test/',
            target_size=(96, 96),
            batch_size=batch_size
        )

        model.evaluate_generator(test_gen)