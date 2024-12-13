# Assignment11

step 1:
Created virtual envirenment with mlflow installed

step 2:
ran mlflow on local host with:

mlflow server --host 127.0.0.1 --port 8080

step 3:
Ran the created python script to create and train the model experiment

python model.py

step 4:
Verify experiment creation in localhost

step 5:
with model uri and other info create a inferencing endpoiont in the original model.py script

import json
import requests

payload = json.dumps(
	{
		"inputs": {
		"feature_1": 0.25,
		"feature_2": 0.520,
		"feature_3": 0.094
			}
		}
	)

requests.post(
	url=f"http://localhost:5000/invocations",
	data=payload,
	headers={"Content-Type": "application/json"}
	)

print(requests.json())

step 6:
deploy the model on inference server (pip intall mlflow[extras]

mlflow models serve -m model:/basic_iris_model/1.0 -p 5000 --enable-mlserver

step 7:
using /inferencing send data to the infrance server 
mlflow models predict -m runs:/basic_iris_model/model -i input.csv -o output.csv
