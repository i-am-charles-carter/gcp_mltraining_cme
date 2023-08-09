# Import libraries
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import keras.metrics
import time


def build_model(X_train, X_val, X_test, Y_train, Y_val, Y_test, class_weights, features):
    """Builds the LSTM model
    Args:
        ds (tuple): A tuple of train_X, train_y, test_X, test_y, class_weights
        features (int): The number of features in the dataset
        Returns:
        history: The model's training history
        """

    num_classes = len(np.unique(Y_train))
    encoder = LabelEncoder()

    # Convert categorical data to numerical data
    train_y_numerical = encoder.fit_transform(Y_train)
    test_y_numerical = encoder.transform(Y_test)

    # Convert numerical labels to binary vectors
    train_y_categorical = to_categorical(train_y_numerical, num_classes=num_classes)
    test_y_categorical = to_categorical(test_y_numerical, num_classes=num_classes)
    print('Building the model')
    # Create the LSTM network
    model = Sequential()
    model.add(LSTM(60, input_shape=(X_train.shape[1], features)))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{time.time()}")

    callbacks = [
        tensorboard
    ]

    # Set class weights
    class_weights_numerical = {encoder.transform([key])[0]: value for key, value in class_weights.items()}

    METRICS = [
        keras.metrics.CategoricalCrossentropy(name='categorical_crossentropy'),
        keras.metrics.MeanSquaredError(name='Brier score'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]
    print('Compiling the model')
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)
    # summarize the model
    model.summary()


    # Fit the model
    history = model.fit(X_train, train_y_categorical,
                        epochs=50,
                        batch_size=512,
                        validation_data=(X_test, test_y_categorical),
                        # class_weight=class_weights_numerical,
                        callbacks=callbacks,
                        # class_weight=class_weights_numerical,
                        verbose=0)
    # evaluate the model
    # _, accuracy = model.evaluate(test_X, test_y_categorical, verbose=0)
    # print('Accuracy: %.2f' % (accuracy*100))

    # evaluate the model using sklearn's metrics
    y_pred = model.predict(X_val).argmax(axis=1)
    y_true = X_val.argmax(axis=1)
    print(classification_report(y_true, y_pred))

    return history, model
