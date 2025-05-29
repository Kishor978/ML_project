from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models():
    """Define and return all model configurations"""
    return  {
            "LogisticRegression": LogisticRegression(
                class_weight='balanced',
                random_state=42,
                C=0.01,
                penalty='l1',
                solver='liblinear',
                max_iter=1000
            ),
            "RandomForest": RandomForestClassifier(
                class_weight='balanced',
                n_estimators=200,
                random_state=42,
                max_depth=None,
                min_samples_leaf=1,
                min_samples_split=2
            ),
            "XGBoost": XGBClassifier(
                scale_pos_weight=1.5,  # Adjust based on class imbalance
                eval_metric='logloss',
                random_state=42,
                learning_rate=0.1,
                n_estimators=200,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                verbosity=0
            ),
            "SVM": SVC(
                kernel='linear',
                class_weight='balanced',
                probability=True,
                random_state=42,
                C=0.1
            ),
            "LightGBM": LGBMClassifier(
                class_weight='balanced',
                random_state=42,
                learning_rate=0.1,
                max_depth=5,
                n_estimators=200,
                verbosity=-1
            )
        }