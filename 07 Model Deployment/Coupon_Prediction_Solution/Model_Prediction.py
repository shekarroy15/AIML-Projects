#importing libraries
#importing the libraries for data analysis
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

campaign_df=pd.read_csv('customer_data.csv')
y = campaign_df['Coupon'].values
campaign_df.drop('Coupon', axis=1, inplace=True)
X=campaign_df.copy()


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# std=StandardScaler()
# X_train=std.fit_transform(X_train)
# X_test=std.fit_transform(X_test)


print("Training model")
reg=RandomForestClassifier()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)

#saving my model
pickle.dump(reg, open('rf_model.pkl','wb'))
print("dumping complete")
#
# #loading the model

model = pickle.load(open('rf_model.pkl','rb'))






