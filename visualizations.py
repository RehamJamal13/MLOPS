import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Convert to DataFrame for easier plotting
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Plot a histogram of a feature (e.g., sepal length)
plt.figure(figsize=(10, 5))
sns.histplot(df['sepal length (cm)'], bins=20, kde=True)
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Plot a scatter plot of two features (e.g., sepal length vs. sepal width)
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='target', palette='Set1')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species', loc='upper left', labels=iris.target_names)
plt.show()

# Train the Logistic Regression model and evaluate performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

# Plot the confusion matrix
plt.figure(figsize=(10, 5))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

