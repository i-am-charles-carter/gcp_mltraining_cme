# from sklearn.metrics import classification_report

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
print('Data loaded successfully')
# Load the model
history, model = build_model(ds, features)


