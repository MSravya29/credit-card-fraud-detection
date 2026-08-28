"""Train the XGBoost fraud model from data/Credit_card.csv.
Run: python train_model.py
"""
from pathlib import Path
import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE=Path(__file__).parent
DATA=BASE/'data'/'Credit_card.csv'
if not DATA.exists(): raise FileNotFoundError('Put Credit_card.csv inside the data folder.')
df=pd.read_csv(DATA); X=df.drop(columns=['Class']); y=df['Class']
scaler=StandardScaler(); X[['Time','Amount']]=scaler.fit_transform(X[['Time','Amount']])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
neg=(ytr==0).sum(); pos=(ytr==1).sum()
model=XGBClassifier(n_estimators=250,max_depth=5,learning_rate=.08,subsample=.85,colsample_bytree=.85,objective='binary:logistic',eval_metric='logloss',scale_pos_weight=neg/pos,random_state=42,n_jobs=4,tree_method='hist')
model.fit(Xtr,ytr); p=model.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int); cm=confusion_matrix(yte,pred)
metrics={'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0),'roc_auc':roc_auc_score(yte,p),'pr_auc':average_precision_score(yte,p),'confusion_matrix':cm.tolist(),'test_size':len(yte),'fraud_count':int(y.sum()),'genuine_count':int((y==0).sum()),'feature_names':X.columns.tolist()}
(BASE/'model').mkdir(exist_ok=True)
joblib.dump(model,BASE/'model'/'fraud_model.pkl',compress=3); joblib.dump(scaler,BASE/'model'/'scaler.pkl')
(BASE/'model'/'metrics.json').write_text(json.dumps(metrics,indent=2))
fi=pd.DataFrame({'Feature':X.columns,'Importance':model.feature_importances_}).sort_values('Importance',ascending=False)
fi.to_csv(BASE/'model'/'feature_importance.csv',index=False)
print(json.dumps(metrics,indent=2))
