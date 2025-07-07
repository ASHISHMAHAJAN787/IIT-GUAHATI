# 📦 Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# 📥 Step 2: Load Data
train = pd.read_csv("Train_Data.csv")
test = pd.read_csv("Test_Data.csv")

# Step 3: Preprocessing Target
train = train.dropna(subset=['age_group'])  # Drop rows with missing target
train['age_group'] = train['age_group'].map({'Adult': 0, 'Senior': 1})  # Encode

# 🧹 Step 4: Handle Missing Values
categorical_cols = ['RIAGENDR', 'PAQ605', 'DIQ010']
numerical_cols = ['BMXBMI', 'LBXGLU', 'LBXGLT', 'LBXIN']
features = categorical_cols + numerical_cols

cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer = SimpleImputer(strategy="median")

train[categorical_cols] = cat_imputer.fit_transform(train[categorical_cols])
train[numerical_cols] = num_imputer.fit_transform(train[numerical_cols])
test[categorical_cols] = cat_imputer.transform(test[categorical_cols])
test[numerical_cols] = num_imputer.transform(test[numerical_cols])

# 🧪 Step 5: EDA (Optional Visualization)
plt.figure(figsize=(6,4))
sns.countplot(x='age_group', data=train)
plt.title("Target Distribution (Adult=0, Senior=1)")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(train[features + ['age_group']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation with age_group")
plt.show()

# 🎯 Step 6: Split Features and Target
X = train[features]
y = train['age_group']

# ⚖️ Step 7: Apply SMOTE to balance the data
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

print("Before SMOTE:", y.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_resampled).value_counts().to_dict())

# 🧠 Step 8: Train a Classifier
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🧪 Step 9: Evaluate on Validation Set
y_pred = model.predict(X_val)
print("\n✅ Accuracy:", accuracy_score(y_val, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_val, y_pred))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 📈 Step 10: Predict on Test Data
X_test = test[features]
test_preds = model.predict(X_test)

# 🔍 Check prediction distribution
print("\n🧮 Test Prediction Counts:", dict(zip(*np.unique(test_preds, return_counts=True))))

# 💾 Step 11: Create Submission File
submission = pd.DataFrame({'age_group': test_preds})
submission.to_csv("submission.csv", index=False)
print("\n✅ submission.csv saved successfully.")
