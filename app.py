from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pycaret.regression import *
import pandas as pd
import numpy as np
import uvicorn
from mangum import Mangum
import ast

app = FastAPI()
handler = Mangum(app)

@app.post('/')
def predict(text: str):

    data = ast.literal_eval(text)
    array = np.array(data)
    loaded_pipeline = load_model('tuned_pipeline')
    feature_names = ['Unnamed: 0', 'crim','zn','chas','nox','rm','age','dis','rad','ptratio','b','lstat']
    index = np.ones(shape=(array.shape[0],1))
    ip = np.concatenate((index, array), axis = -1)
    ip_df = pd.DataFrame(data=ip, columns=feature_names)
    op = predict_model(loaded_pipeline, data=ip_df)
    return JSONResponse({'predictions': op['prediction_label'].to_numpy().tolist()})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9000)
    
# curl -X POST http://0.0.0.0:8080/ \
# -H 'Content-Type: application/json' \
# -d '{[[0.00632,18.0,0,0.538,6.575,65.2,4.09,1,15.3,396.9,4.98]]}'

