import joblib
from typing import Optional
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import config as cfg
import data_manager
import feature_generator
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
def make_prediction(model: str, x, config) -> Optional[tuple[str, float]]:
    # no features → no prediction
    if x is None or len(x) == 0:
        return None
    
    # artifact = joblib.load(model)
    # predictor = artifact['model']
    predictor = xgb.XGBClassifier()
    predictor.load_model(config["model_path_H1"])
    

    pred = predictor.predict(x)
    if pred is None or len(pred) == 0:
        return None
    
    pred_class = pred[0]
    proba_arr = predictor.predict_proba(x)
    if proba_arr is None or len(proba_arr) == 0:
        return None
    
    proba = max(proba_arr[0])

    if pred_class == 0:
        direction = "sell"
    elif pred_class == 1:
        direction = "hold"
    else:
        direction = "buy"

    if direction == "hold" or proba < config['threshold']:
        return None

    return (direction, proba)


def retrain_model(time_frame, config):
    
    end_date = datetime.now() + timedelta(hours=3)
    start_date = end_date - timedelta(days=(config["train_months"]+config["validation_months"]) * 30)
    print(f"Downloading data from {start_date} to {end_date}")
    
    df = data_manager.get_mt5_rates("XAUUSD.ecn", time_frame, "time_range",from_time=start_date, to_time=end_date)
    features = feature_generator.MT5FeaturesManager(df)
    feature = features.add_all_features()
    clean_features = features.handle_missing_values('drop')
    
    if df is None:
        return None, None
  
    val_start = end_date - timedelta(days=config["validation_months"] * 30)
    train_start = val_start - timedelta(days=config["train_months"] * 30)
     
    # Split data
    full_mask = (clean_features.index >= train_start) & (clean_features.index <= end_date) 
    data = clean_features[full_mask].copy()
    y_data_mask = (clean_features.index >= val_start) & (clean_features.index <= end_date)
    y_data = clean_features[y_data_mask].copy()
    
    train_mask = (clean_features.index >= train_start) & (clean_features.index < val_start)
    test_mask = (clean_features.index >= val_start) & (clean_features.index <= end_date)
    train_data = clean_features[train_mask].copy()
    test_data = clean_features[test_mask].copy()

    print(f"Full Data len : {len(df)}\nsplit size : {100 - (len(test_data)/len(train_data))*100:.0f} %")
    print(f"Train : {train_start.strftime('%Y-%m-%d %H:%M:%S')} to {val_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Validation : {val_start.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if len(data) == 0 or len(y_data) == 0:
        print(f"No data available for retraining model")
         
    
    # Prepare features and targets
    labeled = feature_generator.create_target_variable(data,25,'dynamic',1.0)
    
    X = labeled[cfg.MODEL2_FEATURES_H1]
    Y = labeled["label"]
    
    
    X_train, y_train = X.loc[train_data.index],Y.loc[train_data.index]
    X_test, y_test = X.loc[test_data.index], Y.loc[test_data.index]
    
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Not enough data after feature engineering for retraining model")
    
    # Train & prediction & metrics
    print(f"Training on {len(X_train)} samples...")
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train,y_train)
    
    print(f"Predicting on {len(X_test)} validation samples...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    features_importance = model.feature_importances_
    features_names = X.columns
    importance = pd.Series(features_importance,features_names)
    sorted_importance = importance.sort_values(ascending=True)
    print(sorted_importance[::-1])
    # plot is for jupyter run
    # plt.figure(figsize=(10,6))
    # sorted_importance.plot(kind="barh")
    # plt.title("xgboost features")
    # plt.xlabel("importance score")
    # plt.tight_layout()
    # plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred, zero_division=0,digits=2)
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Classification Report :\n{cr}")
    # bundle = {
    #     "model": model,
    #     "features": X,
    #     "conf_threshold": 0.2
    #         }
    model.save_model('model/H1/H1_model.json')
    # joblib.dump(bundle, 'model/M15/M15_model.pkl')
    
    