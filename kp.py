# Essential imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load your Kepler dataset
# Replace with your actual file path
df = pd.read_csv('/kaggle/input/newkeplerdata/keplerdata.csv')

# First look at the data
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
display(df.head()) 
print("\nDataset info:")
df.info()
# Check the distribution of exoplanet classifications
# This is CRUCIAL - we need to know what we're predicting
print("Target variable distribution:")
print(df['koi_disposition'].value_counts())  # Common column name in Kepler data

# Visualize the target distribution
plt.figure(figsize=(10, 6))
df['koi_disposition'].value_counts().plot(kind='bar')
plt.title('Distribution of Exoplanet Classifications')
plt.xlabel('Classification')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show() # Check for missing values
print("Missing values in each column:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

# Check basic statistics
print("\nBasic statistics:")
display(df.describe()) # Common important features in Kepler data for exoplanet detection
important_features = [
    'koi_period',           # Orbital period
    'koi_time0bk',          # Transit epoch
    'koi_impact',           # Impact parameter
    'koi_duration',         # Transit duration
    'koi_depth',            # Transit depth
    'koi_prad',             # Planetary radius
    'koi_teq',              # Equilibrium temperature
    'koi_insol',            # Insolation flux
    'koi_model_snr',        # Signal-to-noise ratio
    'koi_steff',            # Stellar temperature
    'koi_slogg',            # Stellar surface gravity
    'koi_srad'              # Stellar radius
]

# Select features (adjust based on your actual column names)
features = [col for col in important_features if col in df.columns]
print("Available important features:", features)

# Create a working dataframe with selected features
X = df[features].copy()
y = df['koi_disposition'].copy()  # Target variable

# Handle missing values
print("Missing values before handling:")
print(X.isnull().sum())

# Fill missing values with median (you can experiment with different strategies)
X = X.fillna(X.median())

# Verify no missing values remain
print("\nMissing values after handling:")
print(X.isnull().sum().sum()) # Correlation with target (if target is numeric, else use other methods)
plt.figure(figsize=(12, 8))
correlation_with_target = X.copy()
# For visualization, we might need to encode the target temporarily
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

correlation_with_target['target'] = y_encoded
sns.heatmap(correlation_with_target.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')
plt.show() # Encode the target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Target classes mapping:")
for i, class_name in enumerate(le.classes_):
    print(f"{i}: {class_name}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Start with a simple Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
print("Initial Model Performance:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Initial Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show() # See which features are most important
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance for Exoplanet Detection')
plt.tight_layout()
plt.show()

print("Top 5 most important features:")
print(feature_importance.head())
