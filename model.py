import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('salary7.csv')

# ordinal encoding
ordinal_feature = ['Education', 'Age']

edu_ordering = ['Less than a Bachelors', 'Bachelor’s degree', 'Master’s degree', 'Post grad']
age_ordering = ['Prefer not to say', 'Under 18 years old', '18-24 years old', '25-34 years old',
                 '35-44 years old', '45-54 years old', '55-64 years old', '65 years or older']

def ordinal_encode(df, column, ordering):
    df = df.copy()
    df[column] = df[column].apply(lambda x : ordering.index(x))
    return df

df = ordinal_encode(df, 'Education', edu_ordering)
df = ordinal_encode(df, 'Age', age_ordering)

# Onehot Encoding
from sklearn.preprocessing import OneHotEncoder
onehot= OneHotEncoder()
df = pd.get_dummies(df, prefix='', prefix_sep='', columns = ['Country'], drop_first=True)


# train, test split
train, test = train_test_split(df, train_size=0.70, test_size=0.30, random_state=2)

target = 'Salary'
features = train.drop(columns=[target]).columns

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]



# make pipeline with LinearRegression
lm = LinearRegression()


# fitting model with training data
lm.fit(X_train, y_train)

# saving model to disk
pickle.dump(lm, open('model.pkl', 'wb'))

# loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))

