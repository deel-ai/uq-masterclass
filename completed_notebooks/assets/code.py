from deel.puncc.api.prediction import BasePredictor
from deel.puncc.regression import SplitCP

# Define your model (from scikit-learn, TensorFlow, Pytorch, ...)
model = ...

# Wrap the model
predictor = BasePredictor(model)

# Instantiate conformal predictor
splitcp = SplitCP(predictor, train=True)

# Fit model and compute nonconformity scores
splitcp.fit(
    X_fit=X_fit,
    y_fit=y_fit,
    X_calib=X_calib,
    y_calib=y_calib,
)  # or splitcp.fit(X_train, y_train, fit_ratio=0.6)

# Generate prediction intervals for a risk level of 10%
y_pred, y_pred_lower, y_pred_upper = splitcp.predict(X_new, alpha=0.1)


from deel.puncc.api.prediction import BasePredictor
from deel.puncc.classification import APS

# Define and train your model
model = ...

# Wrap your pretrained model
predictor = BasePredictor(model, is_trained=True)

# Instantiate conformal predictor and set "train" to False
aps = APS(predictor, train=False)

# Compute nonconformity scores
aps.fit(X_calib=X_calib, y_calib=y_calib)

# Generate prediction sets for a risk level of 5%
y_pred, set_pred = aps.predict(X_new, alpha=0.05)

from deel.puncc.api.prediction import IdPredictor
from deel.puncc.object_detection import SplitBoxWise


# Initialize a predictor instance to serve as a proxy for the model API
predictor = IdPredictor()

# Set up the conformal predictor
cod = SplitBoxWise(predictor, train=False)

# Compute nonconformity scores for calibration
# `X_calib` should be the prediction output from the API (not raw features), while `y_calib` is the true label for calibration data
cod.fit(X_calib=y_calib_api, y_calib=y_calib_true)

# Predict bounding boxes for new data, and both inner and outer bounding boxes for a specified risk level (alpha) of 30%
y_pred_new, box_inner, box_outer = cod.predict(y_new_api, alpha=0.3)
