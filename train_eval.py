import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
try:
    from imblearn.over_sampling import SMOTE
    _HAS_IMBLEARN = True
except Exception:
    SMOTE = None
    _HAS_IMBLEARN = False
import scipy.stats as stats
from sklearn.utils import resample


PIMA_COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
GBI = ['Glucose','BMI','Insulin']
ALL_FEATS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
```python
import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
import xgboost as xgb
try:
    from imblearn.over_sampling import SMOTE
    _HAS_IMBLEARN = True
except Exception:
    SMOTE = None
    _HAS_IMBLEARN = False
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import StackingClassifier
import warnings
warnings.filterwarnings('ignore')


PIMA_COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
GBI = ['Glucose','BMI','Insulin']
ALL_FEATS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']


def create_advanced_features(df):
    df = df.copy()
    df['BMI_Age_Ratio'] = df['BMI'] / (df['Age'] + 1e-6)
    df['Glucose_BMI_Ratio'] = df['Glucose'] / (df['BMI'] + 1e-6)
    df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1e-6)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0,30,45,60,100], labels=[0,1,2,3]).astype(int)
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0,18.5,25,30,100], labels=[0,1,2,3]).astype(int)
    df['Glucose_Category'] = pd.cut(df['Glucose'], bins=[0,100,126,300], labels=[0,1,2]).astype(int)
    df['Glucose_Age_Interaction'] = df['Glucose'] * df['Age'] / 1000
    df['BMI_Pregnancies_Interaction'] = df['BMI'] * df['Pregnancies']
    df['Metabolic_Risk'] = (df['BMI'] > 30).astype(int) + (df['Glucose'] > 126).astype(int) + (df['BloodPressure'] > 140).astype(int)
    df['Age_Risk'] = (df['Age'] > 45).astype(int)
    # A few extra derived features
    df['Glucose_Squared'] = df['Glucose'] ** 2 / 10000
    df['BMI_Squared'] = df['BMI'] ** 2 / 1000
    df['Log_Insulin'] = np.log1p(df['Insulin'])
    return df


def clean_and_engineer_pima(df, use_advanced_imputation=True):
    df = df.copy()
    zero_as_missing = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for c in zero_as_missing:
        df[c] = df[c].replace(0, np.nan)

    if use_advanced_imputation:
        # Iterative imputation with RandomForestRegressor-like behavior
        from sklearn.ensemble import RandomForestRegressor
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=42), max_iter=10, random_state=42)
        df[zero_as_missing] = imputer.fit_transform(df[zero_as_missing])
    else:
        for c in zero_as_missing:
            df[c] = df[c].fillna(df[c].median())

    df = create_advanced_features(df)
    df = df.dropna()
    return df


def train_ultra(df, feature_set='All'):
    # build feature list
    if feature_set == 'GBI':
        basic_features = GBI
    else:
        basic_features = ALL_FEATS
        engineered_features = [
            'BMI_Age_Ratio','Glucose_BMI_Ratio','Insulin_Glucose_Ratio',
            'Age_Group','BMI_Category','Glucose_Category',
            'Glucose_Age_Interaction','BMI_Pregnancies_Interaction',
            'Metabolic_Risk','Age_Risk','Glucose_Squared','BMI_Squared','Log_Insulin'
        ]
        basic_features = basic_features + engineered_features

    available_features = [f for f in basic_features if f in df.columns]
    X = df[available_features].values
    y = df['Outcome'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # balance train set
    if _HAS_IMBLEARN:
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    else:
        X_train_bal, y_train_bal = X_train, y_train

    # stacking ensemble
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced_subsample', random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, random_state=42, eval_metric='logloss')),
        ('gb', GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.8, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=150, max_depth=10, class_weight='balanced_subsample', random_state=42)),
        ('svm', Pipeline([('scaler', RobustScaler()), ('svm', SVC(kernel='rbf', C=10.0, probability=True, class_weight='balanced', random_state=42))]))
    ]

    meta = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    stack = StackingClassifier(estimators=estimators, final_estimator=meta, cv=5, stack_method='predict_proba', n_jobs=-1)

    stack.fit(X_train_bal, y_train_bal)

    y_pred = stack.predict(X_test)
    y_proba = stack.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(stack, X_train_bal, y_train_bal, scoring='accuracy', cv=cv, n_jobs=-1)

    print(f"Samples after processing: {len(df)}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC AUC: {auc:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"CV Accuracy (train balanced): {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    return acc, auc, f1


if __name__ == '__main__':
    df = pd.read_csv('archive/diabetes.csv')
    df = clean_and_engineer_pima(df, use_advanced_imputation=True)
    acc, auc, f1 = train_ultra(df, feature_set='All')
    if acc >= 0.85:
        print('TARGET_REACHED')
    else:
        print('TARGET_NOT_REACHED')

```
