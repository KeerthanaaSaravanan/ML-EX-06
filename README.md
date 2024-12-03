# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients
<H3>NAME: KEERTHANA S</H3>
<H3>REGISTER NO.: 212223240070</H3>
<H3>EX. NO.6</H3>
<H3>DATE: 23.09.24</H3>

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries.
2. Load the dataset using pd.read_csv().
3. Display data types, basic statistics, and class distributions.
4. Visualize class distributions with a bar plot.
5. Scale feature columns using MinMaxScaler.
6. Encode target labels with LabelEncoder.
7. Split data into training and testing sets with train_test_split().
8. Train LogisticRegression with specified hyperparameters and evaluate the model using metrics and a confusion matrix plot. 

## Program:

```py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/food_items.csv"
data = pd.read_csv(url)

# Display basic info about the dataset
print(data.dtypes,"\n")

# Separate features and target variable
X_raw = data.iloc[:, :-1]  # All columns except the last one (features)
y_raw = data.iloc[:, -1]   # The last column (target variable)

# Feature,target scaling: MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Initialize and train the Logistic Regression model
model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000, multi_class='multinomial', l1_ratio=0.5)
model.fit(X_train, y_train)

# Predict using the model
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


```
## Output:
![image](https://github.com/user-attachments/assets/ff209173-ba0c-4e34-850b-7202d82696a4)
![image](https://github.com/user-attachments/assets/9c76a768-b943-4182-bf01-a0eb4dfb110e)
![image](https://github.com/user-attachments/assets/3fe0a312-182a-4b84-a88c-443205999697)


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
