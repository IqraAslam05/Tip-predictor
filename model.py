# save_model.py
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#  Load the Kaggle 'tips' dataset directly from seaborn
import seaborn as sns
data = sns.load_dataset('tips')

#  Select useful features and target
# We'll predict 'tip' based on total_bill, size, and whether the person is a smoker
data['is_smoker'] = data['smoker'].map({'Yes': 1, 'No': 0})
X = data[['total_bill', 'size', 'is_smoker']]
y = data['tip']

#  Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train model
model = LinearRegression()
model.fit(X_train, y_train)

#  Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(" model.pkl created successfully using the Kaggle 'tips' dataset!")
