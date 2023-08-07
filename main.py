# from sklearn.metrics import classification_report
import comet_ml
from comet_ml import Experiment
experiment = Experiment(
    api_key = "5g0ZjHlM9nzNjpI3sLjAwl6D7",
    project_name = "ml-cme",
    workspace="i-am-charles-carter",
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
)
from get_data import get_data
from build_model import build_model
import tensorflow as tf
import tensorboard

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "window_size":60,
    "features" : 1,
    "y_window_size": 10,
    "test_size" : 0.4,
    "product" : "spx"
}
experiment.log_parameters(hyper_params)

# Load and preprocess data
ds = get_data(product=hyper_params["product"], 
              x_window_size=hyper_params["window_size"], 
              y_window_size=hyper_params["y_window_size"], 
              test_size=hyper_params["test_size"])

with experiment.train():
    build_model(ds, hyper_params['features'])

