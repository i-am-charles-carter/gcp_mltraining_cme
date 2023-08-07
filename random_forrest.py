# def build_random_forrest():
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.estimator import RandomForestClassifier
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.preprocessing import LabelEncoder
# import tensorboard

# # All other methods remains same

# model = RandomForestClassifier(
#     n_estimators=100,  # number of trees in the forest
#     model_dir='.',  
#     feature_columns=feature_columns,  # change this variable to your list of feature names
#     label_vocabulary=['Very Bearish', 'Bearish', 'Flat', 'Bullish', 'Very Bullish'],
#     n_batches_per_layer=1
# )
# train_func = tf.compat.v1.estimator.inputs.numpy_input_fn(
#     x={'x': train_X}, y=train_y, batch_size=512, num_epochs=None, shuffle=True)

# classifier.train(input_fn=train_func, steps=10)

# test_func = tf.compat.v1.estimator.inputs.numpy_input_fn(
#     x={'x': test_X}, y=test_y, num_epochs=1, shuffle=False)

# accuracy_score = classifier.evaluate(input_fn=test_func)['accuracy']
# print('Random Forest has an Accuracy of: {0:f}'.format(accuracy_score))