import mlflow

model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
predictions = model.predict(pd.read_csv("input.csv"))
predictions.to_csv("output.csv")