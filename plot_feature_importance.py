import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# load trained model
model = joblib.load("xgboost_model.pkl")

xgb.plot_importance(model,
                    max_num_features=20,
                    importance_type="gain")

plt.title("Top Important Features - XGBoost")
plt.show()