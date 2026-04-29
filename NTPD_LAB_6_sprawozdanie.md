# Sprawozdanie nr 6
![PBS WTIiE logo CMYK poziom PL-01.png](https://pbs.edu.pl/upload/media/default/0001/01/b6cecfbda538ed95961609f41725eeecb1c0aaff.svg) <br>
**Nazwa ćwiczenia:** CI/CD dla modeli ML – automatyzacja wdrażania <br>
**Przedmiot:** Nowoczesne techniki przetwarzania danych [LAB]<br>
**Student grupa:** Szymon Piórkowski, gr. I<br>
**Data ćwiczeń:** 15.04.2026 r. <br>
**Data oddania sprawozdania:** 29.04.2026 r. <br>

## 1. Cel ćwiczenia

Celem ćwiczenia było nabycie praktycznych umiejętności w zakresie tworzenia i weryfikacji prostego projektu ML z wykorzystaniem testów jednostkowych oraz narzędzi wspierających CI/CD.

## 2. Przebieg ćwiczenia

### **Część właściwa zadania**

### Zadanie 1: Przygotowanie repozytorium z przykładowym modelem ML

1. Utwórz nowy repozytorium na GitHubie o nazwie np. ML-CI-CD. Skorzystaj się z aplikacji API z poprzednich zajęć; 

Utworzono nowe repozytorium na platformie GitHub. Repozytorium zawierać będzie kod aplikacji API wykorzystującej model uczenia maszynowego dla zbioru Iris.

![Zrzut ekranu 2026-04-15 223933.png](Zrzuty%20ekranu/Zrzut%20ekranu%202026-04-15%20223933.png)

2. Napisz jeden/kilka testów jednostkowych dla swojej aplikacji API (za pomocą biblioteki pytest lub unittest):

Przygotowano plik `model.py`, w którym umieszczono funkcje odpowiedzialne za trenowanie modelu i regresji logistycznej oraz wykonanie predykcji. Dodatkowo utworzono funkcję `get_accuracy()`, która oblicza dokładność modelu.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_predict():
    iris = load_iris() # wczytywanie danych
    X = iris.data # cechy
    y = iris.target # klasy

    # podział danych w proporcjach 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    model = LogisticRegression(max_iter=2000) # inicjalizacja modelu

    model.fit(X_train, y_train) # trenowanie modelu
    preds = model.predict(X_test) # predykcja na zbiorze testowym

    return preds, y_test

def get_accuracy():
    preds, y_test = train_and_predict()
    accuracy = accuracy_score(y_test, preds) # obliczanie dokładności
    return accuracy
```

W pliku `test_model.py` przygotowano testy jednostkowe z wykorzystaniem biblioteki `pytest`. Testy sprawdzają, czy model zwraca predykcje, czy liczba predykcji zgadza się z liczbą próbek testowych, czy wartości predykcji mieszczą się w zakresie klas zbioru Iris oraz, czy dokładność modelu wynosi co najmniej 70%.


```python

import numpy as np
from model import train_and_predict, get_accuracy


def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."


def test_predictions_length():
    """
    Test 2 (na maksymalną ocenę 5): Sprawdza, czy długość listy predykcji jest większa od 0 i czy odpowiada
    przewidywanej liczbie próbek testowych.
    """
    preds, y_test = train_and_predict()

    assert len(preds) > 0, "Predictions should not be empty."
    assert len(preds) == len(y_test), "Predictions length should match the number of test samples."


def test_predictions_value_range():
    """
    Test 3 (na maksymalną ocenę 5): Sprawdza, czy wartości w predykcjach mieszczą się w spodziewanym zakresie:
    Dla zbioru Iris mamy 3 klasy (0, 1, 2).
    """
    preds, _ = train_and_predict()
    assert np.all(np.isin(preds, [0, 1, 2])), "Predictions should contain only values 0, 1, or 2."


def test_model_accuracy():
    """
    Test 4 (na maksymalną ocenę 5): Sprawdza, czy model osiąga co najmniej 70% dokładności (przykładowy
    warunek, można dostosować do potrzeb).
    """
    accuracy = get_accuracy()
    assert accuracy >= 0.7, "Model accuracy should be greater than 70."

```

Po uruchomieniu testów lokalnie wszystkie cztery testy zakończyły się sukcesem.  

![Zrzut ekranu 2026-04-28 122147.png](Zrzuty%20ekranu/Zrzut%20ekranu%202026-04-28%20122147.png)

3. Wyślij cały kod aplikacji do repozytorium na GitHubie. Pamiętaj o pliku „requirements.txt”; 

Cały kod aplikacji został przesłany do repozytorium GitHub. 

![Zrzut ekranu 2026-04-28 122553.png](Zrzuty%20ekranu/Zrzut%20ekranu%202026-04-28%20122553.png)

### Zadanie 2: Konfiguracja GitHub Actions do automatycznego testowania 

 
1. W repozytorium na GitHubie utwórz workflow GitHub Actions, który: Będzie uruchamiał się automatycznie przy każdym pushu i pull requeście do gałęzi main; Zainstaluje zależności (z pliku „requirements.txt”); Uruchomi testy jednostkowe; 

Skonfigurowano GitHub Actions. Utworzono plik workflow "python-tests.yml" w katalogu ".github/workflows". Workflow automatycznie uruchamia się po każdym wypchnięciu zmian na gałąż main oraz po utworzeniu pull requesta do tej gałęzi. Workflow pobiera kod repozytorium, konfiguruje środowisko Python, instaluje zależnoścci z pliku `requirements.txt`, a następnie uruchamia testy jednostkowe za pomocą `pytest`.

```yaml
name: Python tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest
```

2. Upewnij się, że workflow poprawnie się wykonuje i raportuje wynik w sekcji „Actions” w GitHubie; 

Po wypchnięciu zmian do repozytorium workflow został uruchomiony automatycznie. W sekcji Actions widoczny jest zakończony proces testowania, co potwierdza poprawną konfigurację automatycznego uruchamiania testów jednostkowych. 


![Zrzut ekranu 2026-04-28 125036.png](Zrzuty%20ekranu/Zrzut%20ekranu%202026-04-28%20125036.png)
![Zrzut ekranu 2026-04-28 125216.png](Zrzuty%20ekranu/Zrzut%20ekranu%202026-04-28%20125216.png)
 


### Zadanie 3: Automatyczne budowanie obrazu Dockera i jego publikacja  


1. Dodaj plik Dockerfile, który pozwoli zbudować obraz Dockera z twoim modelem; 

Przygotowano plik Dockerfile, który buduje obraz kontenera z aplikacją FastAPI. Obraz bazuje na Pythonie 3.11, instaluje zależności z pliku `requirements.txt`, kopiuje kod aplikacji do kontenera i uruchamia serwer uvicorn.

```Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```



2. Skonfiguruj GitHub Actions tak, aby: Budował obraz po każdym pushu na gałąź main; Publikował go do GitHub Container Registry (ghcr.io) lub Docker Hub (wymaga podania w secrets loginu i hasła); 

Utworzono osobny workflow odpowiedzialny za automatyczne budowanie i publikowanie obrazu Dockera. Workflow uruchamia się po każdym push-u na gałąź main, loguje się do GitHub Container Registry, buduje obraz aplikacji i publikuje go w rejestrze. W konfiguracji zastosowaną zmienną `IMAGE_NAME`, aby nazwa obrazu była zapisana małymi literami, co jest wymagane przez Docker.  

```yaml
name: Build and push Docker Image

on:
  push:
    branches: [ "main" ]

jobs:
  docker:
    runs-on: ubuntu-latest

    permissions: 
      contents: read
      packages: write

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
      - name: Logowanie do GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Ustawienie nazwy obrazu
        run: |
          echo "IMAGE_NAME=ghcr.io/${GITHUB_REPOSITORY,,}/iris-api:latest" >> $GITHUB_ENV
      - name: Budowanie obrazu
        run: |
          docker build -t $IMAGE_NAME .
      - name: Push obrazu
        run : |
          docker push $IMAGE_NAME
```

Po wypchnięciu zmian do repozytorium workflow został uruchomiony automatycznie. Proces budowania oraz publikowania obrazu Dockera zakończył się powodzeniem. Obraz został wypchnięty do GitHub Container Registry.

![Zrzut ekranu 2026-04-28 130830.png](Zrzuty%20ekranu/Zrzut%20ekranu%202026-04-28%20130830.png)

![Zrzut ekranu 2026-04-28 130800.png](Zrzuty%20ekranu/Zrzut%20ekranu%202026-04-28%20130800.png)



3. Zweryfikuj, że obraz został poprawnie opublikowany w wybranym rejestrze;

Poprawność publikacji obrazu zweryfikowano w sekcji Packages na GitHubie. Widoczny jest opublikowany pakiet, co potwierdza, że obraz Dockera został prawidłowo przesłany do rejestru. 


![Zrzut ekranu 2026-04-28 131003.png](Zrzuty%20ekranu/Zrzut%20ekranu%202026-04-28%20131003.png)


![Zrzut ekranu 2026-04-28 134514.png](Zrzuty%20ekranu/Zrzut%20ekranu%202026-04-28%20134514.png)


## 3. Wnioski

* Testy jednostkowe są kluczowym elementem projektu, ponieważ pozwalają na szybkie zweryfikowanie poprawności działania modelu i wykrywanie błędów na wczesnym etapie. 
* GitHub Actions znacząco upraszcza proces automatycznego uruchamiania testów przy każdej zmianie w repozytorium.
* Integracja testów z repozytorium zwiększa pewność, że kod pozostaje stabilny przy kolejnych commitach.
* Konteneryzacja aplikacji za pomocą Dockera umożliwia łatwe uruchomienie projektu w dowolnym środowisku.
* Automatyczne budowanie i publikowanie obrazu w GHCR eliminuje potrzebę ręcznego zarządzania wersjami aplikacji. 
* Ćwiczenie pozwoliło zrozumieć znaczenie automatyzacji w procesie wdrażania modeli uczenia maszynowego. 