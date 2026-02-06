# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Load the dataset and separate features X and label y 
2.Divide the dataset into training and testing sets 
3.Fit a multinomial logistic regression model on the training data. 
4.Predict on the test set and calculate accuracy. 
```
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Anikha Pillai
RegisterNumber: 25009524 
*/
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
data = load_iris()
x, y = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


```

## Output:
<img width="198" height="40" alt="image" src="https://github.com/user-attachments/assets/e5d6fe22-5516-466f-95b5-f144aed4e92b" />
<img width="232" height="102" alt="image" src="https://github.com/user-attachments/assets/be826388-eddc-4752-b76e-20436b14d990" />










## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
