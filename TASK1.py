'''CRISP-ML(Q):
    1.a.i. Business problem: Predict The survival rate of titanic ship pssengers
        ii. Business Objectives: Identifay the survival rate
        iii. Business Constraints: Reduce the inaccurate predaction.
        Success Criteria:
        i. Business success criteria: Reduce the inaccurate predaction.
        ii. ML success criteria: Achieve an accuracy of over 95%
        iii. Economic success criteria:
             
    1.b. Data Collection: Bank -> 418 rows, 12 column (11 Inputs and 1 Ouput)
    2. Data Preprocessing - Cleansing & EDA / Descriptive Analytics
    3. Model Building - Experiment with different models alongside Hyperparameter tuning
    4. Evaluation - Not just model evaluation based on accuracy but we also need 
       to evaluate business & economic success criteria
    5. Model Deployment (Flask)
    6. Monitoring & Maintenance (Prediction results to the database - MySQL / MS SQL)'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the Titanic Dataset
data = pd.read_csv(r"C:\Users\91739\OneDrive\Documents\Python Scripts\tested.csv")


print(data.head())

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",# user
                               pw="965877", # passwrd
                               db="titanic_db")) #database


data.to_sql('titanic',con = engine, if_exists = 'replace', index = False)



# Load the Titanic Dataset
df = pd.read_csv(r"C:\Users\91739\OneDrive\Documents\Python Scripts\tested.csv")

# Display the first few rows of the dataset
df.head()


df.tail()


df.describe()

# Display the first few rows of the dataset
df.head()
# Explore and Preprocess Data
df.isnull().sum()

median = df['Age'].median()
print(median)   #36.0


# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)

# Check if 'Cabin' column exists before dropping
if 'Cabin' in df.columns:
    df.drop(['Cabin'], axis=1, inplace=True)  # Drop 'Cabin' column due to many missing values

# Convert categorical variables to numerical
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)  # Note the change in case for 'embarked'


# ### AutoEDA
##############
# sweetviz
##########
import sweetviz
my_report = sweetviz.analyze([data, "data"])
my_report.show_html('Report1.html')

# D-Tale
########
import dtale
d = dtale.show(data)
d.open_browser()


# Feature selection
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']  # Note the change in case for 'embarked'
X = df[features]
y = df['Survived']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
#Accuracy: 0.73

print("Classification Report:")


conf_matrix = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

## plot graph for survier
pclass_sur = data[['Pclass','Survived']].groupby('Pclass').sum()
pclass_sur

pclass_sur.plot(kind = 'bar')
plt.title('survivers per class')
plt.ylabel('survived number')

sex_sur = data[['Sex', 'Survived']].groupby('Sex').sum()
sex_sur

sex_sur.plot(kind = 'bar')
plt.title('Sex survived per class')
plt.ylabel('survived number')





