import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Creating a simple DataFrame
data = {
    'Feature1': [2, 3, 10, 6, 8, 11, 5, 9],
    'Feature2': [3, 6, 7, 8, 3, 9, 4, 10],
    'Label': [0, 0, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Splitting features and labels
X = df[['Feature1', 'Feature2']]  # Features
y = df['Label']  # Labels

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Support Vector Classifier (SVC)
svm = SVC(kernel='linear')

# Train the model
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of the SVM model: {accuracy * 100:.2f}%")

#To verify the output
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})

#Testing with new data 
#step 1: creating a new data frame
data = {
    'Feature1': [2,13,21],
    'Feature2': [3,16,27],
           }

# Create DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

#step 2: predict with our model
y_pred = svm.predict(df)
print(y_pred)