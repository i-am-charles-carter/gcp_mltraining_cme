import comet_ml
from comet_ml import Experiment

experiment = Experiment(
    api_key="5g0ZjHlM9nzNjpI3sLjAwl6D7",
    project_name="ml-cme",
    workspace="i-am-charles-carter",
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
)

from build_model import build_model
import tensorboard
from tick_to_ohlcv import get_ohlcv
from get_data_tick import prepare_data, shuffle_and_split_data

# Set the parameters
window_size = 180
features = 1
y_window_size = 10
test_size = 0.4
product = "spx"

data = get_ohlcv()
print("Hello Commencing Computeration")
X, Y = prepare_data(data, 60, 5)
X_train, X_val, X_test, Y_train, Y_val, Y_test, class_weights = shuffle_and_split_data(X, Y)
print('Data loaded successfully')

with experiment.train():
    build_model(X_train, X_val, X_test, Y_train, Y_val, Y_test, class_weights, features)
