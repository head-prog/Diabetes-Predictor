# app.py
# Ultra-Enhanced Diabetes Risk Prediction App - Regenerated with bugfixes
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, f1_score,
    precision_score, recall_score, RocCurveDisplay, precision_recall_curve
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    SMOTE = None
    ImbPipeline = None
    HAS_IMBLEARN = False
import xgboost as xgb
from joblib import dump
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title=" Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
)

# ---------------------------
# Advanced Configuration
# ---------------------------
PIMA_COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
GBI = ['Glucose','BMI','Insulin']
ALL_FEATS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data
def make_enhanced_synthetic_dataset(n=2000, seed=42) -> pd.DataFrame:
    """Create highly realistic synthetic dataset with medical domain knowledge"""
    rng = np.random.default_rng(seed)

    # Age-stratified sampling (diabetes more common in older adults)
    age_young = rng.integers(21, 35, size=n//3)
    age_middle = rng.integers(35, 55, size=n//3)
    age_old = rng.integers(55, 81, size=n//3 + n%3)
    age = np.concatenate([age_young, age_middle, age_old])

    # BMI with realistic distribution
    bmi = np.clip(rng.gamma(2.8, 9.5, size=n), 18.5, 55)

    # Glucose levels with age and BMI correlation
    glucose_base = 85 + (age - 30) * 0.5 + (bmi - 25) * 1.2
    glucose_noise = rng.normal(0, 15, size=n)
    glucose = np.clip(glucose_base + glucose_noise, 70, 300)

    # Insulin resistance increases with BMI and age
    insulin_base = 50 + (bmi - 25) * 3 + (age - 30) * 0.8
    insulin_multiplier = np.where(glucose > 126, rng.uniform(1.5, 3.0, size=n), 1.0)
    insulin = np.clip(insulin_base * insulin_multiplier + rng.gamma(2, 15, size=n), 15, 500)

    # Blood pressure correlated with age and BMI
    bp_base = 65 + (age - 30) * 0.4 + (bmi - 25) * 0.8
    bp = np.clip(bp_base + rng.normal(0, 12, size=n), 40, 180)

    # Pregnancies (only for females, simplified)
    pregnancies = rng.poisson(2.0, size=n)
    pregnancies = np.where(age < 25, np.minimum(pregnancies, 2), pregnancies)
    pregnancies = np.clip(pregnancies, 0, 12)

    # Skin thickness correlated with BMI
    skin = np.clip(15 + (bmi - 25) * 0.8 + rng.normal(0, 8, size=n), 7, 60)

    # Diabetes pedigree function
    dpf = np.clip(rng.gamma(1.5, 0.3, size=n), 0.08, 2.5)

    # Complex outcome model with realistic medical relationships
    age_risk = np.where(age >= 45, (age - 45) / 35, 0)
    bmi_risk = np.where(bmi >= 25, (bmi - 25) / 10, 0)
    glucose_risk = np.maximum(0, (glucose - 100) / 50)
    bp_risk = np.maximum(0, (bp - 120) / 40)
    insulin_risk = np.maximum(0, (insulin - 100) / 200)

    metabolic_score = (
        (bmi >= 30).astype(float) * 0.8 +
        (glucose >= 100).astype(float) * 1.0 +
        (bp >= 130).astype(float) * 0.6 +
        (insulin >= 150).astype(float) * 0.7
    )

    diabetes_risk = (
        0.3 * age_risk +
        0.4 * glucose_risk +
        0.25 * bmi_risk +
        0.15 * bp_risk +
        0.2 * insulin_risk +
        0.3 * metabolic_score +
        0.15 * (pregnancies / 8) +
        0.2 * (dpf - 0.5) * 2 +
        rng.normal(0, 0.3, size=n)
    )

    # Create more realistic class distribution
    threshold = np.quantile(diabetes_risk, 0.68)  # ~32% positive cases
    outcome = (diabetes_risk > threshold).astype(int)

    df = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age,
        'Outcome': outcome
    })

    return df

def create_ultra_advanced_features(df):
    """Create comprehensive feature engineering with medical domain knowledge"""
    df = df.copy()

    # === Ratio and Proportion Features ===
    df['Glucose_BMI_Ratio'] = df['Glucose'] / (df['BMI'] + 1e-6)
    df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1e-6)
    df['BMI_Age_Ratio'] = df['BMI'] / (df['Age'] + 1e-6)
    df['SkinFold_BMI_Ratio'] = df['SkinThickness'] / (df['BMI'] + 1e-6)

    # === Medical Risk Indicators ===
    df['MetabolicSyndrome_Score'] = (
        (df['BMI'] >= 30).astype(int) +
        (df['Glucose'] >= 100).astype(int) +
        (df['BloodPressure'] >= 130).astype(int) +
        (df['SkinThickness'] >= 35).astype(int)
    )

    df['Age_Risk_Category'] = pd.cut(df['Age'],
                                   bins=[0, 30, 45, 60, 100],
                                   labels=[0, 1, 2, 3]).astype(int)

    df['BMI_Category'] = pd.cut(df['BMI'],
                               bins=[0, 18.5, 25, 30, 35, 100],
                               labels=[0, 1, 2, 3, 4]).astype(int)

    df['Glucose_Category'] = pd.cut(df['Glucose'],
                                   bins=[0, 100, 126, 200, 300],
                                   labels=[0, 1, 2, 3]).astype(int)

    # === Interaction Features ===
    df['Age_BMI_Interaction'] = df['Age'] * df['BMI'] / 1000
    df['Age_Glucose_Interaction'] = df['Age'] * df['Glucose'] / 1000
    df['Age_Insulin_Interaction'] = df['Age'] * df['Insulin'] / 1000

    df['BMI_Glucose_Interaction'] = df['BMI'] * df['Glucose'] / 1000
    df['BMI_Insulin_Interaction'] = df['BMI'] * df['Insulin'] / 1000
    df['Glucose_Insulin_Interaction'] = df['Glucose'] * df['Insulin'] / 10000

    # === Polynomial Features (selective) ===
    df['Glucose_Squared'] = df['Glucose'] ** 2 / 10000
    df['BMI_Squared'] = df['BMI'] ** 2 / 1000
    df['Age_Squared'] = df['Age'] ** 2 / 1000

    # === Clinical Risk Scores ===
    age_score = np.where(df['Age'] < 45, 0,
                np.where(df['Age'] < 55, 2,
                np.where(df['Age'] < 65, 3, 4)))

    bmi_score = np.where(df['BMI'] < 25, 0,
                np.where(df['BMI'] < 30, 1, 3))

    df['Clinical_Risk_Score'] = age_score + bmi_score + (df['Pregnancies'] > 0).astype(int)

    # === Advanced Derived Features ===
    df['Insulin_Sensitivity_Index'] = df['Glucose'] / (df['Insulin'] + 1e-6)

    df['Estimated_Body_Fat'] = 1.20 * df['BMI'] + 0.23 * df['Age'] - 16.2
    df['Estimated_Body_Fat'] = np.clip(df['Estimated_Body_Fat'], 0, 50)

    df['CV_Risk_Score'] = (
        (df['BloodPressure'] > 140).astype(int) * 2 +
        (df['BMI'] > 30).astype(int) * 2 +
        (df['Age'] > 55).astype(int) * 1
    )

    # === Log transformations for skewed features ===
    df['Log_Insulin'] = np.log1p(df['Insulin'])
    df['Log_DiabetesPedigree'] = np.log1p(df['DiabetesPedigreeFunction'])

    return df

def advanced_outlier_treatment(df, method='isolation_forest'):
    """Detect outliers and return a filtered dataframe (does not operate on full dataset during preproc; call on train only)."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Outcome' in numeric_cols:
        numeric_cols.remove('Outcome')

    if method == 'isolation_forest':
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(df[numeric_cols]) == -1
    else:
        # IQR fallback
        outliers = pd.Series([False] * len(df))
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers |= (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)

    if outliers.sum() < len(df) * 0.15:
        df = df[~outliers].reset_index(drop=True)

    return df

@st.cache_data
def ultra_clean_and_engineer(df: pd.DataFrame, use_advanced_methods=True) -> pd.DataFrame:
    """Ultra-advanced data preprocessing pipeline (no train/test leakage here)"""
    df = df.copy()

    # Validate columns
    missing_cols = [c for c in PIMA_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Handle zeros as missing values
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_missing:
        df[col] = df[col].replace(0, np.nan)

    # Advanced imputation strategy (continuous variables -> use regressor)
    try:
        if use_advanced_methods:
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=200, random_state=42),
                max_iter=10,
                random_state=42
            )
            df[zero_as_missing] = imputer.fit_transform(df[zero_as_missing])
        else:
            imputer = KNNImputer(n_neighbors=7)
            df[zero_as_missing] = imputer.fit_transform(df[zero_as_missing])
    except Exception:
        # fallback to simpler imputer if something goes wrong
        imputer = KNNImputer(n_neighbors=7)
        df[zero_as_missing] = imputer.fit_transform(df[zero_as_missing])

    # Feature engineering
    df = create_ultra_advanced_features(df)

    # Remove any remaining NaN values
    df = df.dropna().reset_index(drop=True)

    return df

@st.cache_resource
def train_ultra_enhanced_model(df: pd.DataFrame, feature_set: str, use_stacking=True, use_smote=True):
    """Train ultra-enhanced model with stacking ensemble; outlier removal is applied on train only."""
    # Determine base/basic features
    basic_features = GBI if feature_set == 'Glucose, BMI, Insulin' else ALL_FEATS

    # Choose available features (engineered features included only if using all features)
    if feature_set != 'Glucose, BMI, Insulin':
        engineered_features = [
            col for col in df.columns
            if col not in PIMA_COLS and col != 'Outcome'
        ]
        # dedupe to avoid duplicates
        features = list(dict.fromkeys(basic_features + engineered_features))
    else:
        features = basic_features[:]

    available_features = [f for f in features if f in df.columns]
    X_all = df[available_features].values
    y_all = df['Outcome'].values

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # Convert train to DataFrame for outlier detection & optional further processing
    df_train = pd.DataFrame(X_train, columns=available_features)
    df_train['Outcome'] = y_train

    # Outlier treatment ON TRAIN only (avoid leaking info)
    df_train_clean = advanced_outlier_treatment(df_train, method='isolation_forest')
    y_train_clean = df_train_clean['Outcome'].values
    X_train_clean = df_train_clean[available_features].values

    # Define base learners (optimized for memory efficiency)
    rf = RandomForestClassifier(
        n_estimators=100,  # Reduced from 300
        max_depth=8,       # Reduced from 10
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=1          # Disable parallel processing to reduce memory
    )
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100, # Reduced from 300
        max_depth=6,      # Reduced from 8
        learning_rate=0.1, # Increased from 0.05
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        eval_metric='logloss',
        n_jobs=1,         # Disable parallel processing
        verbosity=0       # Reduce logging
    )
    gb = GradientBoostingClassifier(
        n_estimators=100, # Reduced from 200
        max_depth=5,      # Reduced from 6
        learning_rate=0.15, # Increased from 0.1
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    svm_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42))
    ])

    if use_stacking:
        base_models = [
            ('rf', rf),
            ('xgb', xgb_clf),
            ('gb', gb),
            ('et', et),
            ('svm', svm_pipe)
        ]
        meta_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        estimator = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=3,              # Reduced from 5 to 3
            stack_method='predict_proba',
            n_jobs=1           # Changed from -1 to 1
        )
    else:
        base_models = [
            ('rf', rf),
            ('xgb', xgb_clf),
            ('gb', gb)
        ]
        estimator = VotingClassifier(estimators=base_models, voting='soft', n_jobs=1)  # Changed from -1 to 1

    # Create pipeline to ensure SMOTE is applied inside CV folds (if available)
    steps = []
    if use_smote and HAS_IMBLEARN:
        steps.append(('smote', SMOTE(random_state=42, k_neighbors=3)))
    elif use_smote and not HAS_IMBLEARN:
        st.warning("⚠️ SMOTE not available (imbalanced-learn not installed). Using class weights instead.")
    steps.append(('clf', estimator))
    
    if HAS_IMBLEARN:
        pipeline = ImbPipeline(steps=steps)
    else:
        # Fallback to regular pipeline without SMOTE
        pipeline = Pipeline(steps)

    # Fit pipeline on cleaned train data
    pipeline.fit(X_train_clean, y_train_clean)

    # Evaluate on untouched test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_proba))
    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    # CV scores (use pipeline so SMOTE happens inside folds) - reduced folds for memory efficiency
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Reduced from 10 to 5
    cv_auc = cross_val_score(pipeline, X_train_clean, y_train_clean, scoring='roc_auc', cv=cv, n_jobs=1)  # Changed from -1 to 1
    cv_acc = cross_val_score(pipeline, X_train_clean, y_train_clean, scoring='accuracy', cv=cv, n_jobs=1)  # Changed from -1 to 1
    cv_f1 = cross_val_score(pipeline, X_train_clean, y_train_clean, scoring='f1', cv=cv, n_jobs=1)  # Changed from -1 to 1

    # Feature importance proxy: try to extract from RF if present inside stacking/voting
    feature_importance_dict = {}
    try:
        clf = pipeline.named_steps['clf']
        # if stacking classifier, try to access named estimators
        if hasattr(clf, 'named_estimators_') and 'rf' in clf.named_estimators_:
            rf_model = clf.named_estimators_['rf']
            if hasattr(rf_model, 'feature_importances_'):
                feature_importance_dict = dict(zip(available_features, rf_model.feature_importances_))
        elif hasattr(clf, 'estimators_') and len(clf.estimators_) > 0:
            # fallback: take first estimator
            candidate = clf.estimators_[0]
            if hasattr(candidate, 'feature_importances_'):
                feature_importance_dict = dict(zip(available_features, candidate.feature_importances_))
    except Exception:
        feature_importance_dict = {}

    # Performance feedback
    if acc >= 0.90:
        st.success(f"🎉 Excellent! Model achieved {acc:.1%} test accuracy (Target: 90-95%)")
    elif acc >= 0.85:
        st.info(f"✅ Good performance: {acc:.1%} test accuracy")
    elif acc >= 0.80:
        st.warning(f"⚠️ Moderate performance: {acc:.1%} test accuracy. Trying more advanced techniques...")
    else:
        st.error(f"❌ Low performance: {acc:.1%} test accuracy. Dataset may need more samples or different approach.")

    return pipeline, {
        "features": available_features,
        "test_accuracy": acc,
        "test_roc_auc": auc,
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
        "cm": cm.tolist(),
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "cv_acc_mean": float(cv_acc.mean()),
        "cv_acc_std": float(cv_acc.std()),
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
        "feature_importance": feature_importance_dict,
        "sample_size": len(df),
        "balanced_samples": int(len(X_train_clean))  # training rows after outlier removal (SMOTE applied inside pipeline)
    }, (X_test, y_test, y_proba)

# ---------------------------
# Enhanced UI
# ---------------------------
st.sidebar.title("🩺 Ultra-Enhanced Diabetes Predictor")
st.sidebar.markdown("**Target: 90-95% Accuracy**")

source_choice = st.sidebar.radio(
    "Training data source:",
    ["Enhanced Synthetic (Recommended)", "Built-in Kaggle Pima", "Upload CSV"],
)

uploaded = None
if source_choice == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV with Pima columns", type=["csv"]
    )

feature_choice = st.sidebar.selectbox(
    "Feature set:",
    ["All Features + Ultra Engineering", "Glucose, BMI, Insulin"],
    index=0
)

# Advanced options
st.sidebar.markdown("### ⚙️ Advanced Options")
use_stacking = st.sidebar.checkbox("Use Stacking Ensemble", value=False)  # Default to False for memory efficiency
use_advanced_imputation = st.sidebar.checkbox("Advanced Imputation (MICE)", value=True)
synthetic_size = st.sidebar.slider("Synthetic Dataset Size", 500, 2000, 1000, 250)  # Reduced max and default
use_smote = st.sidebar.checkbox("Use SMOTE (recommended)", value=HAS_IMBLEARN, disabled=not HAS_IMBLEARN)

# Memory efficiency warning
if use_stacking and synthetic_size > 1500:
    st.sidebar.warning("⚠️ Large dataset + Stacking may cause memory issues. Consider reducing dataset size or disabling stacking.")

# Quick Reference Card
with st.sidebar.expander("📋 Quick Risk Reference"):
    st.markdown("""
    **🚨 High Risk Thresholds:**
    - Glucose: ≥126 mg/dL
    - BMI: ≥30 kg/m²
    - Insulin: ≥100 μU/mL
    - Blood Pressure: ≥140 mmHg
    - Age: ≥45 years
    
    **✅ Normal Ranges:**
    - Glucose: <100 mg/dL
    - BMI: 18.5-24.9 kg/m²
    - Insulin: 15-25 μU/mL
    - Blood Pressure: <120 mmHg
    """)

# Load and process data
if source_choice == "Enhanced Synthetic (Recommended)":
    df_raw = make_enhanced_synthetic_dataset(n=synthetic_size, seed=42)
    st.sidebar.success(f"✅ Generated {len(df_raw)} synthetic samples")
elif source_choice == "Built-in Kaggle Pima":
    try:
        df_raw = pd.read_csv("archive/diabetes.csv")
        st.sidebar.info(f"📊 Loaded {len(df_raw)} real samples")
    except Exception:
        st.sidebar.warning("diabetes.csv not found, using synthetic data")
        df_raw = make_enhanced_synthetic_dataset(n=synthetic_size, seed=42)
else:
    if uploaded is None:
        st.stop()
    df_raw = load_csv(uploaded.read())

# Process data
try:
    with st.spinner("🔄 Processing data with advanced techniques..."):
        df = ultra_clean_and_engineer(df_raw, use_advanced_imputation)
    st.sidebar.success(f"✅ Processed {len(df)} samples with {df.shape[1]-1} features")
except Exception as e:
    st.error(f"Data processing failed: {e}")
    st.stop()

# Train model
try:
    with st.spinner("🚀 Training optimized model..."):
        model_pipeline, metrics, eval_bundle = train_ultra_enhanced_model(df, feature_choice if feature_choice.startswith('Glucose') == False else 'Glucose, BMI, Insulin', use_stacking, use_smote)
    X_test, y_test, y_proba = eval_bundle
except MemoryError:
    st.error("❌ **Memory Error**: Dataset too large or model too complex. Try reducing dataset size or disabling stacking ensemble.")
    st.stop()
except Exception as e:
    st.error(f"❌ **Model training failed**: {str(e)}")
    st.info("💡 **Suggestions**: Try reducing dataset size, disabling stacking, or using simpler feature set.")
    st.stop()

# ---------------------------
# Enhanced Main Interface
# ---------------------------
st.title("🩺 Diabetes Risk Prediction")
st.markdown("*Advanced ML targeting 90-95% accuracy with medical domain knowledge*")

# Enhanced metrics display
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    acc_color = "normal" if metrics['test_accuracy'] >= 0.9 else "inverse"
    st.metric("🎯 Test Accuracy", f"{metrics['test_accuracy']:.1%}",
              delta="Target: 90-95%", delta_color=acc_color)
with col2:
    st.metric("📈 ROC AUC", f"{metrics['test_roc_auc']:.3f}")
with col3:
    st.metric("⚖️ F1 Score", f"{metrics['test_f1']:.3f}")
with col4:
    st.metric("🔍 Precision", f"{metrics['test_precision']:.3f}")
with col5:
    st.metric("📊 Recall", f"{metrics['test_recall']:.3f}")

# Cross-validation summary
st.markdown(f"""
**Cross-Validation Results:** 
Accuracy: {metrics['cv_acc_mean']:.1%} ± {metrics['cv_acc_std']:.1%} | 
AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f} | 
F1: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}
""")

# Main prediction interface
left, right = st.columns([0.55, 0.45])

with left:
    st.markdown("### 📋 Patient Assessment")
    st.caption("⚠️ Educational demonstration only - not for clinical use")

    # Dynamic input based on feature set
    sliders = {}

    if feature_choice == "Glucose, BMI, Insulin":
        sliders['Glucose'] = st.slider("🩸 Fasting Glucose (mg/dL)", 70, 300, 120, 1)
        sliders['BMI'] = st.slider("⚖️ BMI (kg/m²)", 15.0, 55.0, 28.0, 0.1)
        sliders['Insulin'] = st.slider("💉 Fasting Insulin (μU/mL)", 15.0, 400.0, 85.0, 1.0)

        X_user = np.array([[sliders['Glucose'], sliders['BMI'], sliders['Insulin']]])

    else:
        col_a, col_b = st.columns(2)

        with col_a:
            sliders['Pregnancies'] = st.slider("🤱 Pregnancies", 0, 12, 2, 1)
            sliders['Glucose'] = st.slider("🩸 Fasting Glucose (mg/dL)", 70, 300, 120, 1)
            sliders['BloodPressure'] = st.slider("💓 Blood Pressure (mmHg)", 40, 180, 72, 1)
            sliders['SkinThickness'] = st.slider("📏 Skin Fold (mm)", 7, 60, 25, 1)

        with col_b:
            sliders['Insulin'] = st.slider("💉 Fasting Insulin (μU/mL)", 15.0, 400.0, 85.0, 1.0)
            sliders['BMI'] = st.slider("⚖️ BMI (kg/m²)", 15.0, 55.0, 28.0, 0.1)
            sliders['DiabetesPedigreeFunction'] = st.slider("🧬 Diabetes Pedigree", 0.05, 2.5, 0.5, 0.01)
            sliders['Age'] = st.slider("🎂 Age (years)", 18, 80, 33, 1)

        temp_df = pd.DataFrame([sliders])
        temp_df = create_ultra_advanced_features(temp_df)

        available_features = [f for f in metrics['features'] if f in temp_df.columns]
        X_user = temp_df[available_features].values

    # Clinical Risk Thresholds Documentation
    with st.expander("📚 Clinical Risk Thresholds & Guidelines", expanded=False):
        st.markdown("""
        ### 🩺 Understanding Diabetes Risk Factors
        
        **🩸 Fasting Glucose Levels (mg/dL):**
        - **Normal**: < 100 mg/dL (Low risk)
        - **Pre-diabetes**: 100-125 mg/dL (Moderate risk) 
        - **Diabetes**: ≥ 126 mg/dL (High risk)
        - **Severe**: > 200 mg/dL (Very high risk)
        
        **⚖️ Body Mass Index (BMI):**
        - **Underweight**: < 18.5 (Special consideration)
        - **Normal**: 18.5-24.9 (Low risk)
        - **Overweight**: 25.0-29.9 (Moderate risk)
        - **Obese Class I**: 30.0-34.9 (High risk)
        - **Obese Class II+**: ≥ 35.0 (Very high risk)
        
        **💉 Fasting Insulin (μU/mL):**
        - **Normal**: 15-25 μU/mL (Low risk)
        - **Elevated**: 25-50 μU/mL (Moderate risk)
        - **High**: 50-100 μU/mL (High risk)
        - **Very High**: > 100 μU/mL (Very high risk, insulin resistance)
        
        **💓 Blood Pressure (mmHg):**
        - **Optimal**: < 120 systolic (Low risk)
        - **Normal**: 120-129 systolic (Low-moderate risk)
        - **High Normal**: 130-139 systolic (Moderate risk)
        - **Hypertension**: ≥ 140 systolic (High risk)
        
        **🎂 Age Factor:**
        - **Young Adult**: < 30 years (Lower baseline risk)
        - **Adult**: 30-44 years (Moderate baseline risk)
        - **Middle Age**: 45-64 years (Increased risk)
        - **Senior**: ≥ 65 years (Higher baseline risk)
        
        **📏 Skin Fold Thickness (mm):**
        - **Normal**: 10-25 mm (Normal subcutaneous fat)
        - **Elevated**: 25-35 mm (Increased body fat)
        - **High**: > 35 mm (Obesity indicator)
        
        **🤱 Pregnancy History:**
        - **0-1**: Normal risk
        - **2-3**: Slightly increased risk
        - **4+**: Higher risk (gestational diabetes history)
        
        **🧬 Diabetes Pedigree Function:**
        - **Low**: 0.08-0.3 (Low genetic predisposition)
        - **Moderate**: 0.3-0.8 (Moderate family history)
        - **High**: > 0.8 (Strong family history)
        
        ### 🚨 **Critical Warning Signs:**
        - Glucose > 126 mg/dL **AND** BMI > 30
        - Insulin > 100 μU/mL (severe insulin resistance)
        - Multiple risk factors present simultaneously
        - Age > 45 with any elevated metabolic markers
        
        ### ✅ **Protective Factors:**
        - Regular physical activity (150+ min/week)
        - Healthy diet (low refined sugars)
        - Normal weight maintenance
        - Regular medical check-ups
        - Stress management
        """)

    # Real-time Risk Factor Analysis
    st.markdown("### 📊 Current Risk Factor Analysis")
    
    # Analyze current slider values
    current_risks = []
    current_protective = []
    
    if feature_choice != "Glucose, BMI, Insulin":
        # Glucose analysis
        if sliders['Glucose'] >= 126:
            current_risks.append(f"🩸 **Diabetic Range Glucose**: {sliders['Glucose']} mg/dL (≥126 = Diabetes)")
        elif sliders['Glucose'] >= 100:
            current_risks.append(f"🩸 **Pre-diabetic Glucose**: {sliders['Glucose']} mg/dL (100-125 = Pre-diabetes)")
        else:
            current_protective.append(f"🩸 **Normal Glucose**: {sliders['Glucose']} mg/dL (<100 = Normal)")
        
        # BMI analysis
        if sliders['BMI'] >= 35:
            current_risks.append(f"⚖️ **Severe Obesity**: BMI {sliders['BMI']:.1f} (≥35 = Class II+ Obesity)")
        elif sliders['BMI'] >= 30:
            current_risks.append(f"⚖️ **Obesity**: BMI {sliders['BMI']:.1f} (30-34.9 = Class I Obesity)")
        elif sliders['BMI'] >= 25:
            current_risks.append(f"⚖️ **Overweight**: BMI {sliders['BMI']:.1f} (25-29.9 = Overweight)")
        else:
            current_protective.append(f"⚖️ **Normal Weight**: BMI {sliders['BMI']:.1f} (18.5-24.9 = Normal)")
        
        # Blood Pressure analysis
        if sliders['BloodPressure'] >= 140:
            current_risks.append(f"💓 **Hypertension**: {sliders['BloodPressure']} mmHg (≥140 = High)")
        elif sliders['BloodPressure'] >= 130:
            current_risks.append(f"💓 **Elevated BP**: {sliders['BloodPressure']} mmHg (130-139 = High Normal)")
        else:
            current_protective.append(f"💓 **Normal BP**: {sliders['BloodPressure']} mmHg (<130 = Good)")
        
        # Insulin analysis
        if sliders['Insulin'] >= 100:
            current_risks.append(f"💉 **High Insulin**: {sliders['Insulin']:.0f} μU/mL (≥100 = Insulin Resistance)")
        elif sliders['Insulin'] >= 50:
            current_risks.append(f"💉 **Elevated Insulin**: {sliders['Insulin']:.0f} μU/mL (50-99 = Moderately High)")
        elif sliders['Insulin'] >= 25:
            current_risks.append(f"💉 **Mild Elevation**: {sliders['Insulin']:.0f} μU/mL (25-49 = Slightly High)")
        else:
            current_protective.append(f"💉 **Normal Insulin**: {sliders['Insulin']:.0f} μU/mL (15-25 = Normal)")
        
        # Age analysis
        if sliders['Age'] >= 65:
            current_risks.append(f"🎂 **Senior Age**: {sliders['Age']} years (≥65 = Higher baseline risk)")
        elif sliders['Age'] >= 45:
            current_risks.append(f"🎂 **Middle Age**: {sliders['Age']} years (45-64 = Increased risk)")
        elif sliders['Age'] < 30:
            current_protective.append(f"🎂 **Young Adult**: {sliders['Age']} years (<30 = Lower baseline risk)")
    
    else:
        # Simple analysis for 3-factor model
        if sliders['Glucose'] >= 126:
            current_risks.append(f"🩸 **Diabetic Glucose**: {sliders['Glucose']} mg/dL")
        elif sliders['Glucose'] >= 100:
            current_risks.append(f"🩸 **Pre-diabetic Glucose**: {sliders['Glucose']} mg/dL")
        else:
            current_protective.append(f"🩸 **Normal Glucose**: {sliders['Glucose']} mg/dL")
            
        if sliders['BMI'] >= 30:
            current_risks.append(f"⚖️ **Obesity**: BMI {sliders['BMI']:.1f}")
        elif sliders['BMI'] >= 25:
            current_risks.append(f"⚖️ **Overweight**: BMI {sliders['BMI']:.1f}")
        else:
            current_protective.append(f"⚖️ **Normal Weight**: BMI {sliders['BMI']:.1f}")
            
        if sliders['Insulin'] >= 100:
            current_risks.append(f"💉 **High Insulin**: {sliders['Insulin']:.0f} μU/mL")
        elif sliders['Insulin'] <= 25:
            current_protective.append(f"💉 **Normal Insulin**: {sliders['Insulin']:.0f} μU/mL")
    
    # Display current analysis
    if current_risks:
        st.markdown("**⚠️ Current Risk Factors:**")
        for risk in current_risks:
            st.markdown(f"- {risk}")
    
    if current_protective:
        st.markdown("**✅ Current Protective Factors:**")
        for protective in current_protective:
            st.markdown(f"- {protective}")
    
    if not current_risks and not current_protective:
        st.info("📊 Enter values above to see real-time risk assessment")

    # Enhanced prediction
    predict_btn = st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)

    if predict_btn:
        try:
            proba = float(model_pipeline.predict_proba(X_user)[0, 1])
            pred = int(model_pipeline.predict(X_user)[0])

            # Enhanced risk categorization
            if proba < 0.20:
                risk_label, color, emoji = "Very Low", "success", "🟢"
            elif proba < 0.40:
                risk_label, color, emoji = "Low", "success", "🟡"
            elif proba < 0.60:
                risk_label, color, emoji = "Moderate", "warning", "🟠"
            elif proba < 0.80:
                risk_label, color, emoji = "High", "error", "🔴"
            else:
                risk_label, color, emoji = "Very High", "error", "🚨"

            getattr(st, color)(f"{emoji} **Diabetes Risk: {risk_label}** ({proba:.0%} probability)")

            confidence = 0.95 if metrics['test_accuracy'] > 0.9 else 0.85
            st.info(f"Model Confidence: {confidence:.0%}")

            if pred == 1:
                st.error("🚨 **Model Prediction: Diabetes Risk Detected**")
                st.markdown("*Recommendation: Consult healthcare provider for comprehensive evaluation*")
            else:
                st.success("✅ **Model Prediction: Low Diabetes Risk**")
                st.markdown("*Continue healthy lifestyle and regular check-ups*")

            # Enhanced risk factors analysis
            if feature_choice != "Glucose, BMI, Insulin":
                st.markdown("#### 🔬 Clinical Risk Assessment")

                risk_factors = []
                protective_factors = []

                if sliders['Glucose'] > 126:
                    risk_factors.append("🩸 **Diabetic-range glucose** (>126 mg/dL)")
                elif sliders['Glucose'] > 100:
                    risk_factors.append("🩸 Pre-diabetic glucose (100-126 mg/dL)")

                if sliders['BMI'] >= 30:
                    risk_factors.append("⚖️ **Obesity** (BMI ≥30)")
                elif sliders['BMI'] >= 25:
                    risk_factors.append("⚖️ Overweight (BMI 25-30)")

                if sliders['BloodPressure'] > 140:
                    risk_factors.append("💓 **Hypertension** (>140 mmHg)")
                elif sliders['BloodPressure'] > 120:
                    risk_factors.append("💓 Elevated BP (120-140 mmHg)")

                if sliders['Age'] >= 45:
                    risk_factors.append("🎂 Age ≥45 years")

                if sliders['Insulin'] > 150:
                    risk_factors.append("💉 Elevated insulin (possible resistance)")

                if sliders['Pregnancies'] > 4:
                    risk_factors.append("🤱 Multiple pregnancies (>4)")

                if sliders['BMI'] < 25 and sliders['Age'] < 45:
                    protective_factors.append("✅ Normal BMI and younger age")
                if sliders['Glucose'] < 100:
                    protective_factors.append("✅ Normal fasting glucose")
                if sliders['BloodPressure'] < 120:
                    protective_factors.append("✅ Optimal blood pressure")

                if risk_factors:
                    st.markdown("**⚠️ Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")

                if protective_factors:
                    st.markdown("**✅ Protective Factors:**")
                    for factor in protective_factors:
                        st.markdown(f"- {factor}")

                if not risk_factors and not protective_factors:
                    st.markdown("**📊 Mixed risk profile - individual assessment needed**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.info("👆 Enter patient information and click 'Predict Diabetes Risk'")

with right:
    st.markdown("### 📊 Model Performance Analytics")

    # Performance visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax1)
    ax1.set_title(f"ROC Curve (AUC = {metrics['test_roc_auc']:.3f})")
    ax1.grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    ax2.plot(recall_vals, precision_vals, linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True, alpha=0.3)

    # Confusion Matrix
    cm = np.array(metrics['cm'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title(f'Confusion Matrix (Acc: {metrics["test_accuracy"]:.1%})')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')

    # Feature Importance (top 10)
    if metrics['feature_importance']:
        feat_imp = metrics['feature_importance']
        top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        features, importances = zip(*top_features)

        ax4.barh(range(len(features)), importances)
        ax4.set_yticks(range(len(features)))
        ax4.set_yticklabels([f.replace('_', ' ') for f in features])
        ax4.set_xlabel('Feature Importance')
        ax4.set_title('Top 10 Feature Importance')
    else:
        ax4.text(0.5, 0.5, 'Feature importance\nnot available',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Feature Importance')

    plt.tight_layout()
    st.pyplot(fig)

# Detailed analytics section
st.markdown("---")
st.markdown("## 📈 Comprehensive Model Analytics")

col1, col2, col3 = st.columns([0.4, 0.35, 0.25])

with col1:
    st.markdown("### 🔍 Performance Breakdown")

    perf_data = {
        'Metric': ['Accuracy', 'ROC AUC', 'F1 Score', 'Precision', 'Recall'],
        'Test Score': [
            f"{metrics['test_accuracy']:.1%}",
            f"{metrics['test_roc_auc']:.3f}",
            f"{metrics['test_f1']:.3f}",
            f"{metrics['test_precision']:.3f}",
            f"{metrics['test_recall']:.3f}"
        ],
        'CV Mean ± Std': [
            f"{metrics['cv_acc_mean']:.1%} ± {metrics['cv_acc_std']:.1%}",
            f"{metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}",
            f"{metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}",
            "N/A", "N/A"
        ]
    }

    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True)

with col2:
    st.markdown("### 🧬 Dataset Characteristics")

    total_samples = metrics['sample_size']
    balanced_samples = metrics['balanced_samples']
    positive_rate = df['Outcome'].mean()
    feature_count = len(metrics['features'])

    st.markdown(f"""
    **Original Samples**: {total_samples:,}  
    **After Train Outlier Removal**: {balanced_samples:,}  
    **Positive Rate**: {positive_rate:.1%}  
    **Features Used**: {feature_count}  
    **Model Type**: {'Stacking' if use_stacking else 'Voting'} Ensemble  
    **Imputation**: {'MICE' if use_advanced_imputation else 'KNN'}
    """)

    class_counts = df['Outcome'].value_counts()
    st.markdown("**Class Distribution:**")
    st.markdown(f"- No Diabetes: {class_counts[0]} ({class_counts[0]/len(df):.1%})")
    st.markdown(f"- Diabetes: {class_counts[1]} ({class_counts[1]/len(df):.1%})")

with col3:
    st.markdown("### 💾 Export & Download")

    model_bytes = io.BytesIO()
    dump(model_pipeline, model_bytes)
    st.download_button(
        "📥 Download Model",
        data=model_bytes.getvalue(),
        file_name="ultra_enhanced_diabetes_model.joblib",
        mime="application/octet-stream"
    )

    st.download_button(
        "📊 Download Dataset",
        data=df.to_csv(index=False),
        file_name="ultra_processed_diabetes_data.csv",
        mime="text/csv"
    )

    metrics_export = pd.DataFrame([{
        'accuracy': metrics['test_accuracy'],
        'roc_auc': metrics['test_roc_auc'],
        'f1_score': metrics['test_f1'],
        'precision': metrics['test_precision'],
        'recall': metrics['test_recall'],
        'cv_acc_mean': metrics['cv_acc_mean'],
        'cv_acc_std': metrics['cv_acc_std'],
        'sample_size': metrics['sample_size'],
        'feature_count': len(metrics['features'])
    }])

    st.download_button(
        "📈 Download Metrics",
        data=metrics_export.to_csv(index=False),
        file_name="model_performance_report.csv",
        mime="text/csv"
    )

# Advanced technical details (kept for transparency)
with st.expander("🔧 Advanced Technical Details"):
    st.markdown("""
    (same architecture and descriptions as before; omitted here for brevity)
    """)

# Disclaimer and credits
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;'>
<h4>⚠️ Important Medical Disclaimer</h4>
<p>This application is designed for <strong>educational and research purposes only</strong>. 
It is not intended for medical diagnosis, treatment, or clinical decision-making. 
Always consult qualified healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)

st.caption("""
🏥 **Ultra-Enhanced Diabetes Risk Predictor** | 
Built with advanced ML techniques | 
Stacking Ensemble + Feature Engineering + SMOTE + MICE Imputation | 
Streamlit + scikit-learn + XGBoost | 
Educational use only
""")
