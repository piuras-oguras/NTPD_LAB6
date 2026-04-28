from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn
import os

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

app = FastAPI() # utworzenie aplikacji FastAPI

iris = load_iris() # wczytanie zbioru danych Iris
# cechy (X) i etykiety (y)
X = iris.data
y = iris.target

model = LogisticRegression() # utworzenie modelu regresji logistycznej
model.fit(X, y) # trenowanie modelu na danych

# klasa określająca dane wejściowe
class IrisInput(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

# endpoint testowy
@app.get("/")
def root():
    return {"message": "Hello World"}

# endpoint informacyjny zwraca informacje o modelu i danych
@app.get("/info")
def info():
    return {
        'model_type': 'Logistic Regression',
        'features': iris.feature_names,
        'classes': iris.target_names.tolist()
    }

@app.get("/config")
def config():
    return {
        "app_mode": os.getenv("APP_MODE", "default"),
        "model_version": os.getenv("MODEL_VERSION", "v1"),
    }

# endpoint sprawdzający stan API
@app.get("/health")
def health():
    return {"status": "ok"}
# endpoint predykcyjny
@app.post("/predict")
def predict(iris: IrisInput):
    try:
        # pobieranie danych i zapis do listy
        input_data = [iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]
        # walidacja danych — czy wartości są w odpowiednim zakresie
        if not all(0 <= v <= 10 for v in input_data):
            raise HTTPException(status_code=400, detail="Wartości poza zakresem")
        # konwersja danych do formatu NumPy
        input_data = np.array([input_data])
        # wykonanie predykcji
        prediction = model.predict(input_data)[0]
        # mapowanie numerów klas na nazwy
        class_names = ["setosa", "versicolor", "virginica"]
        return {"prediction": class_names[prediction]}
    # obsługa błędów serwera
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd serwera{e}")

# uruchomienie aplikacji
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)