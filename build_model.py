# Import libraries
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import keras.metrics

# Build the model
def build_model(ds, features):
    """Builds the LSTM model
    Args:
        ds (tuple): A tuple of train_X, train_y, test_X, test_y, class_weights
        features (int): The number of features in the dataset
        Returns:
        history: The model's training history
        """
    train_X, test_X, train_y, test_y, class_weights = ds
    num_classes = len(np.unique(train_y))
    encoder = LabelEncoder()

    # Convert categorical data to numerical data
    train_y_numerical = encoder.fit_transform(train_y)
    test_y_numerical = encoder.transform(test_y)

    # Convert numerical labels to binary vectors
    train_y_categorical = to_categorical(train_y_numerical, num_classes=num_classes)
    test_y_categorical = to_categorical(test_y_numerical, num_classes=num_classes)
    print('Building the model')
    # Create the LSTM network
    model = Sequential()
    model.add(LSTM(60, input_shape=(train_X.shape[1], features)))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./logs")

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
        keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    print('Compiling the model')
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)
    # summarize the model
    model.summary()

    # Fit the model
    history = model.fit(train_X, train_y_categorical, 
            epochs=10,
            batch_size=512, 
            validation_data=(test_X, test_y_categorical), 
            class_weight=class_weights_numerical,
            callbacks=callbacks)
    # evaluate the model
    _, accuracy = model.evaluate(test_X, test_y_categorical, verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))

    # evaluate the model using sklearn's metrics
    y_pred = model.predict(test_X).argmax(axis=1)
    y_true = test_X.argmax(axis=1)
    print(classification_report(y_true, y_pred))
    
    return history, model


