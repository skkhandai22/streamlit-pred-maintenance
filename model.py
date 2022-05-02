import pandas as pd
from imblearn.over_sampling import SMOTENC
import sklearn as sk
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle

data = pd.read_csv('predictive_maintenance.csv')

data = pd.read_csv('predictive_maintenance.csv')

data = data.drop(["UDI",'Product ID'],axis=1)

data['Failure Type'] = data['Failure Type'].map({'Heat Dissipation Failure':2, 'No Failure': 1,'Overstrain Failure':3,'Power Failure':4,'Random Failures':5,'Tool Wear Failure':6})
data['Type']= data['Type'].map({'L':1,'M':2,'H':3})

seed = 1
X = data[['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']]
Y = data['Failure Type']
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X,Y,test_size =0.25,random_state=seed)
X_val, X_test, y_val, y_test = sk.model_selection.train_test_split(X_test,y_test,test_size =0.25,random_state=seed)

smote_nc = SMOTENC(categorical_features=[0], random_state=0,sampling_strategy='all')
X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)

df_resampled = pd.concat([X_resampled,y_resampled],axis=1)

LSVC = make_pipeline(StandardScaler(),
                     SVC(random_state=0, tol=1e-5,kernel='rbf'))

LSVC.fit(X_resampled, y_resampled)

LSVC.score(X_resampled, y_resampled)*100


pickle_out = open("model.pkl", mode = "wb")
pickle.dump(LSVC, pickle_out)
pickle_out.close()