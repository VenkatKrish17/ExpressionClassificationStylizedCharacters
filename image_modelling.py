
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,MaxPooling1D,AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from keras.layers import LSTM
from sklearn.metrics import roc_curve
from keras.preprocessing.image import ImageDataGenerator
import keras.optimizers
import matplotlib.pyplot as plt
from keras.utils import plot_model

import numpy as np
import math
from keras.preprocessing import image
import matplotlib.pyplot as plt
input_layer=Input(shape=(64,64,3))
def generate_data_set():
    print("creating data set")
    train_datagen = ImageDataGenerator(rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True
    )
    training_set = train_datagen.flow_from_directory('./preprocessed',
    target_size = (64, 64),
    batch_size = 46169,

    class_mode = 'categorical',
    shuffle=True)
    test_datagen = ImageDataGenerator(rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2
    )
    test_set = test_datagen.flow_from_directory('./test_set',
    target_size = (64, 64),
    batch_size = 10003,

    class_mode = 'categorical')
    # print("returning")
    # print(training_set)
    # print(testing_set)
    print("extracting from data gen")
    rgb_train_x,rgb_train_y= training_set.next()
    rgb_test_x,rgb_test_y=test_set.next()
    print("returning")
    np.save('rgb_train_x.npy',rgb_train_x)
    np.save('rgb_train_y.npy',rgb_train_y)
    np.save('rgb_test_x.npy',rgb_test_x)
    np.save('rgb_test_y.npy',rgb_test_y)
    return (rgb_train_x,rgb_train_y,rgb_test_x,rgb_test_y)
def first_model_cnn(rgb_train_x,weights_file=None):
    input  = input_layer
    #classifier = Sequential()
    # classifier.add(Conv2D(64, (3, 3), input_shape = (64, 64,1), activation = 'relu'))
    # classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # classifier.add(Dropout(0.5))
    # classifier.add(Conv2D(64, (3, 3), input_shape = (64, 64,1), activation = 'relu'))
    # classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # classifier.add(Dropout(0.5))
    layer = Conv2D(64, (3, 3), activation = 'relu')(input)
    layer = MaxPooling2D(pool_size = (2, 2))(layer)

    layer = Conv2D(64, (3, 3), activation = 'relu')(layer)
    layer = MaxPooling2D(pool_size = (2, 2))(layer)

    layer = Conv2D(64, (3, 3), activation = 'relu')(layer)
    layer = MaxPooling2D(pool_size = (2, 2))(layer)
    #classifier = Conv2D(64, (3, 3), input_shape = (64, 64,1), activation = 'relu')(classifier)
    layer = Flatten()(layer)
    # layer.add(Conv2D(16, (3, 3), input_shape = (64, 64,1), activation = 'relu'))
    # classifier.add(MaxPooling2D(pool_size = (2, 2)))
    #classifier.add(Flatten())
    # Step 4 - Full connection
    #64 gave 100%
    #layer = Dense(units = 64, activation = 'softmax') (layer)
    #layer= Dense(units = 32, activation = 'softmax')(layer)
    output = Dense(units = 7, activation = 'softmax') (layer)
    # classifier.add(Dense(units = 64, activation = 'softmax'))
    # classifier.add(Dense(units = 7, activation = 'softmax'))
    #classifier.summary()
    classifier = Model(input,output)
    classifier.summary()
    if(weights_file!=None):
        classifier.load_weights(weights_file,by_name=True)
    optimizer_new=keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier
def second_model_rnn(rgb_train_x,rgb_train_y,rgb_test_x,rgb_test_y,weights_file=None):
    row, col, pixel = rgb_train_x.shape[1:]
    row_hidden = 64
    col_hidden = 64
    x = Input(shape=(row, col, pixel))
    # Encodes a row of pixels using TimeDistributed Wrapper.
    encoded_rows = TimeDistributed(LSTM(row_hidden))(x)
    # Encodes columns of encoded rows.
    encoded_columns = LSTM(col_hidden)(encoded_rows)
    # Final predictions and model.
    #encoded_columns=MaxPooling1D()(encoded_columns)
    #encoded_columns = Flatten()(col_hidden)(encoded_rows)
    inner_layer = Dense(128, activation='relu')(encoded_columns)
    prediction = Dense(7, activation='softmax')(encoded_columns)
    model = Model(x,prediction)
    #model.add(AveragePooling2D(pool_size = (2, 2)))
    #model.add(Flatten())
    #optimizer_new=keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()

    return model
def npcnn_model(rgb_train_x,weights_file=None):
    input  = input_layer
    #classifier = Sequential()
    layer = Conv2D(64, (3, 3), activation = 'relu')(input)
    layer =Conv2D(64, (3, 3), activation = 'relu')(layer)
    layer =Conv2D(64, (3, 3), activation = 'relu')(layer)

    #classifier.add(Conv2D(16, (3, 3), activation = 'relu'))
    #classifier.add(Flatten())
    layer = Flatten()(layer)
    #classifier.add(Dense(units = 7, activation = 'softmax'))
    #layer = Dense(units = 32, activation = 'softmax')(layer)
    output = Dense(units = 7, activation = 'softmax')(layer)
    classifier = Model(input,output)
    classifier.summary()
    if(weights_file!=None):
        classifier.load_weights(weights_file,by_name=True)
    #optimizer_new=keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier
def mlp_model_mlp(rgb_train_x,rgb_train_y,rgb_test_x,rgb_test_y,weights_file=None):
    print("constructing the third model")
    inputs = input_layer
    x = Dense(128, activation='relu')(inputs)
    #x = Dropout(1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    #x = Dropout(1)(x)
    x = Flatten()(x)
    predictions = Dense(units=7,activation='softmax')(x)
    #print("prd")
    model = Model(inputs=inputs, outputs=predictions)
    #print("mod")
    model.summary()
    if(weights_file!=None):
        model.load_weights(weights_file,by_name=True)
    optimizer_new=keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
    model.compile(optimizer=optimizer_new,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model  # starts training
def fit_model(classifier,training_set,training_labels,file_destination,epochs_count):
    try:
        filepath=file_destination
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]
        history=classifier.fit(x=training_set,
        y=training_labels,
        epochs = epochs_count,
        validation_split=0.3,
        callbacks=callbacks_list
        )
        print(history.history.keys())
        print(history.history)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        classifier.save('/modelpath')
    except Exception as e:
        print(e)
def evaluate_model(model,rgb_test_x,rgb_test_y):
    print("Evaluating model")
    pred =model.predict(rgb_test_x)
    t_y_max=rgb_test_y.argmax(axis=1)
    p_y_max=pred.argmax(axis=1)
    confusion_matrix = metrics.confusion_matrix(t_y_max, p_y_max)
    print(confusion_matrix)
    print(metrics.accuracy_score(t_y_max, p_y_max, normalize=True))
    print(metrics.classification_report(t_y_max, p_y_max))
    #roc_curve(t_y_max, p_y_max)
def build_first_model(use_weights):
    #rgb_train_x,rgb_train_y,rgb_test_x,rgb_test_y=generate_data_set()
    rgb_train_x=np.load('rgb_train_x.npy')
    rgb_train_y=np.load('rgb_train_y.npy')
    rgb_test_x=np.load('rgb_test_x.npy')
    rgb_test_y=np.load('rgb_test_y.npy')
    if(use_weights):
        model=first_model_cnn(rgb_train_x,'weights_first_model_rgb.hdf5')
    else:
        model=first_model_cnn(rgb_train_x)
        plot_model(model, to_file='first_model_cnn.png')
        fit_model(model,rgb_train_x,rgb_train_y,'weights_first_model_rgb.hdf5',5)
        print("Evaluating first model")
        evaluate_model(model,rgb_test_x,rgb_test_y)
    plot_model(model, to_file='first_model_cnn.png')
    return model
def build_mlp_model(use_weights):
    rgb_train_x=np.load('rgb_train_x.npy')
    rgb_train_y=np.load('rgb_train_y.npy')
    rgb_test_x=np.load('rgb_test_x.npy')
    rgb_test_y=np.load('rgb_test_y.npy')
    if(use_weights):
        model=mlp_model_mlp(rgb_train_x,rgb_train_y,rgb_test_x,rgb_test_y,'weights_mlp_final_rgb.hdf5')
    else:
        model=mlp_model_mlp(rgb_train_x,rgb_train_y,rgb_test_x,rgb_test_y)
        fit_model(model,rgb_train_x,rgb_train_y,'weights_mlp_final_rgb.hdf5',5)
        print("Evaluating third model")
        evaluate_model(model,rgb_test_x,rgb_test_y)
    plot_model(model, to_file='mlp_model.png')

    return model
def build_second_model(use_weights):
    rgb_train_x=np.load('rgb_train_x.npy')
    rgb_train_y=np.load('rgb_train_y.npy')
    rgb_test_x=np.load('rgb_test_x.npy')
    rgb_test_y=np.load('rgb_test_y.npy')
    if(use_weights):
        model=second_model_rnn(rgb_train_x,rgb_train_y,rgb_test_x,rgb_test_y,'weights_lstm_second_rgb.hdf5')
    else:
        model=second_model_rnn(rgb_train_x,rgb_train_y,rgb_test_x,rgb_test_y)
        fit_model(model,rgb_train_x,rgb_train_y,'weights_lstm_second_rgb.hdf5',12)
        print("Evaluating second model")
        evaluate_model(model,rgb_test_x,rgb_test_y)
    #model.summary()
    plot_model(model, to_file='second_model_rnn.png')
    return model
def build_npcnn_model(use_weights):
    rgb_train_x=np.load('rgb_train_x.npy')
    rgb_train_y=np.load('rgb_train_y.npy')
    rgb_test_x=np.load('rgb_test_x.npy')
    rgb_test_y=np.load('rgb_test_y.npy')
    if(use_weights):
        model=npcnn_model(rgb_train_x,'weights_no_pool_cnn_rgb.hdf5')
    else:
        model=npcnn_model(rgb_train_x)
        fit_model(model,rgb_train_x,rgb_train_y,'weights_no_pool_cnn_rgb.hdf5',5)
        print("Evaluating npcnn model")
        evaluate_model(model,rgb_test_x,rgb_test_y)
    #model.summary()
    plot_model(model, to_file='npcnn_model.png')
    return model

def ensemble_models(models_array,rgb_train_x):
    outputs = [model.outputs[0] for model in models_array]
    #print(outputs)
    rgb_train_x=np.load('rgb_train_x.npy')
    rgb_train_y=np.load('rgb_train_y.npy')
    rgb_test_x=np.load('rgb_test_x.npy')
    rgb_test_y=np.load('rgb_test_y.npy')
    y = keras.layers.Average()(outputs)
    # print(rgb_train_x.shape)
    #print(y)
    input  = Input(shape=models_array[0].input.shape[1:])
    model = Model(input_layer, y, name='ensemble')
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()
    fit_model(model,rgb_train_x,rgb_train_y,'ensemble_weights_rgb.hdf5',2)
    print("Evaluating ensembled model")
    evaluate_model(model,rgb_test_x,rgb_test_y)
    plot_model(model, to_file='ensemble_model.png')
    return model

def custom_transfer_learning(weights_file=None):
    model = keras.applications.VGG19(weights = "imagenet", include_top=False,pooling=(2,2), input_shape=(64,64,3))
    #model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', pooling=(2,2), classes=7)
    x = Flatten()(model(input_layer))
    x = Dense(7, activation='relu')(x)
    model = Model(inputs=input_layer, outputs=x)
    optimizer_new=keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
    if(weights_file!=None):
        model.load_weights(weights_file)
    model.compile(optimizer=optimizer_new,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    plot_model(model, to_file='transfer_learning.png')
    model.summary()
    return model


def build_custom_model(use_weights):
    rgb_train_x=np.load('rgb_train_x.npy')
    rgb_train_y=np.load('rgb_train_y.npy')
    rgb_test_x=np.load('rgb_test_x.npy')
    rgb_test_y=np.load('rgb_test_y.npy')
    if(use_weights):
        model=custom_transfer_learning('weight_transfer_rgb.hdf5')
    else:
        model=custom_transfer_learning()
        fit_model(model,rgb_train_x,rgb_train_y,'weight_transfer_rgb.hdf5',6)
        print("Evaluating transfer model")
        evaluate_model(model,rgb_test_x,rgb_test_y)
    return model

    pass
    #model.summary()

try:
    #generate_data_set()
    rgb_train_x=np.load('rgb_train_x.npy')
    rgb_test_x=np.load('rgb_test_x.npy')
    rgb_test_y=np.load('rgb_test_y.npy')
    first_model=build_first_model(use_weights=True)
    #second_model=build_second_model(use_weights=False)
    custom_model=build_custom_model(use_weights=True)
    #evaluate_model(second_model,rgb_test_x,rgb_test_y)
    npcnn_model=build_npcnn_model(use_weights=True)
    #mlp_model=build_mlp_model(use_weights=True)
    #evaluate_model(mlp_model,rgb_test_x,rgb_test_y)

    # #evaluate_model(first_model,rgb_test_x,rgb_test_y)
    #evaluate_model(npcnn_model,rgb_test_x,rgb_test_y)
    # custom_model=build_custom_model(use_weights=True)
    # evaluate_model(custom_model,rgb_test_x,rgb_test_y)
    #evaluate_model
    ensemble=ensemble_models([first_model,custom_model,npcnn_model],rgb_train_x)
    #custom_transfer_learning(use_weights=None)

    #evaluate_model(ensemble,rgb_test_x,rgb_test_y)
except Exception as e:
    print(e)
