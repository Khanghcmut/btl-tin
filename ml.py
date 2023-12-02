print("python run")

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.saving import load_model


def resnes(data_path,out_path):
    
    scaler = StandardScaler()
    
    path=os.path.dirname("ml.py")
    
    file='matlab_tinhocvlkt.csv'
    full_csv=os.path.join(path,file)
    full_model=os.path.join(path,'./model_resnes.h5')

    out_path_data=os.path.join(path,out_path)

    out=pd.read_csv(full_csv)
    model=load_model(full_model)
    out=out.iloc[1:, 1:-1]
    emg=pd.read_csv(data_path)

    true_move=emg['restimulus'].values
    emg=emg.iloc[:,1:-1]
    emglen=len(emg)
    out=pd.concat([out,emg],axis=0)
    
    
    X_test =out.values
    
    
    X_test = scaler.fit_transform(X_test)
    X_test=X_test[-emglen:]

    predictionDL_test=model.predict(X_test)
    pred_test = np.argmax(predictionDL_test, axis=1)
    # print(pred_test[-1])
    emg['Predict out']=pred_test
    emg['restimulus']=true_move
    emg.to_csv(out_path_data)
    return out_path_data
result=resnes(data_path,out_path)
# resnes([9.391486,   6.706713 ,  2.289105,  1.649242,  2.174856, 2,4,27,14,26,30,7,3.961060,   3.850974,   9.939316,  13.807969])
# resnes([4,	11.3,	3.,	2.79,	3.429,	29.792,	15.5600,	9.78774744,	17.21714,	7.8313,	4.005931220,
#    2.02670,	3.234192325759250,	5.5767439480,	8,	20.098])#expect11

