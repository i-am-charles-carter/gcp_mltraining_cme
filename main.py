from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf
from get_data import get_data
from build_model import build_model
import tensorboard

# Set the parameters
window_size = 60
features = 1
y_window_size = 10
test_size = 0.4
product = "spx"

# Load and preprocess data
ds = get_data(product=product, 
              x_window_size=window_size, 
              y_window_size=y_window_size, 
              test_size=test_size)

# Load the model
history, model = build_model(ds, features)

# evaluate the model
_, accuracy = model.evaluate(ds['test_X'], ds['test_y'])
print('Accuracy: %.2f' % (accuracy*100))

# evaluate the model using sklearn's metrics
y_pred = model.predict(ds['test_X']).argmax(axis=1)
y_true = ds['test_X'].argmax(axis=1)
print(classification_report(y_true, y_pred))

# plotting results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss.png')