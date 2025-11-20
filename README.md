# Vehicle-Accident-Predictor
- Paso 1: crear el grid espacio-temporal para discretizar el tiempo y espacio. No podemos predecir un accidente en toda la AP-7 hay que hacerlo por tramos (definidos a cada 10km). En este grid contiene cada hora de cada segmento de km que tenemos, y con una columna Y_ACCIDENT (incialmente en 0), recorremos el csv de accidentes y ponemos 1 en la fila correspondiente del grid que hemos creado. 
- Paso 2: feature engineering
    - feature categoricas -> one-hot encoding ej. D_TIPUS_VIA: [1,0,0] Autopista
- Paso 3: Creación de Secuencias (Preparación para LSTM)
Tu modelo predecirá el riesgo en la hora T basándose en las últimas N horas.

Define tu "Lookback" (Ventana): Decide cuántas horas atrás mirará el modelo (ej. N = 12 horas).

Genera las Secuencias: Debes "trocear" tu "Grid" con una ventana deslizante. Cada "muestra" de entrenamiento será:

X (Features): Un tensor 3D de forma (N_horas, N_features). Por ejemplo, (12, 50) si usas 12 horas de lookback y tienes 50 features (clima, hora_sin, hora_cos, etc.).

Y (Objetivo): Un escalar: 0 o 1 (el valor de Y_ACCIDENT en la hora T+1).

Manejo del Desbalanceo: Al entrenar, usa class_weights en Keras/TensorFlow para dar mil veces más importancia a las muestras "1" que a las "0".

- Paso 4: Construcción del Modelo LSTM. Una arquitectura robusta podría ser:

Loss: binary_crossentropy (perfecto para clasificación 0/1).

Metrics: NO uses 'accuracy' (será 99.99%). Usa 'Precision', 'Recall' y 'AUC-PR' (Area Under Precision-Recall Curve). Tu objetivo es maximizar el Recall (encontrar los accidentes) manteniendo una Precision aceptable

# OUTPUT TRAIN MEJOR OBTENIDO HASTA AHORA
Loading data...
Data split: Train size: 3527913 Test size: 881979
Training with 3527913 rows...
Ratio base: 7735.65
Starting hyperparameter search (20 combinations)...
Fitting 3 folds for each of 20 candidates, totalling 60 fits
Best hyperparameters found:
{'subsample': 0.8, 'scale_pos_weight': np.float64(15471.302631578947), 'reg_lambda': 1.0, 'reg_alpha': 1.0, 'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.01, 'colsample_bytree': 0.7}
Best AUC in validation: {search.best_score_:.4f}
Training XGBoost...

 Evaluating...
 ROC AUC Final Score: 0.6327
 Selected threshold: 0.311818 (for Recall ~55.00000000000001%)

--- FINAL RESULTS ---

 Confusion matrix:
[[590640 291265]
 [    34     40]]

Report:
              precision    recall  f1-score   support

           0       1.00      0.67      0.80    881905
           1       0.00      0.54      0.00        74

    accuracy                           0.67    881979
   macro avg       0.50      0.61      0.40    881979
weighted avg       1.00      0.67      0.80    881979


Most Important Variables:
precipitation          0.173157
temperature            0.072502
D_TRACAT_ALTIMETRIC    0.070923
hour_sin               0.062649
dow_cos                0.061797
humidity               0.059587
segmento_pk            0.058679
dow_sin                0.055592
month_cos              0.054278
month_sin              0.054183

## Notebook Overview-Data
Este cuaderno tiene por objetivo conocer los datos y tener una primera visión de los datasets AP7 y catalunya, para poder crear un conjunto de datos capaz de alimentar el modelo de prediccion (hacer feautre engineering)-
Qué se consigue:
- Explorar e integrar las fuentes (accidents, clima, incidències).  
- Generar visualitzacions i mètriques preliminars per avaluar la rellevància de les variables i la qualitat del dataset.  

## Flow - Steps
- [x]  Obtener datos: historicos + contextuales + tiempo real (meteocat + vehiculos?)
- [ ]  Preprocesamiento
    - [x]  Limpiar datos — filtrar
    - [x]  Entender los campos y hacer una visualización inicial
    - [x]  Feature engineering — para que el modelo entienda las caracteristicas de texto --> LABEL ENCODING, NO one hot encoding pq queda el csv muy sucio
    - [x]  Conseguir datos del tiempo
- [ ]  Hacer el modelo
    - [ ]  LSTM
    - [ ]  XGBOOST
    - [ ]  Test y validación del modelo.
- [ ]  Especificar el funcionamiento en la nube
- [ ]  App + dashboard


## Referencies
dades historiques per entrenar el model:
https://datos.gob.es/ca/catalogo/a09002970-accidentes-de-trafico-con-fallecidos-o-heridos-graves-en-cataluna

dades contextuals: https://datos.gob.es/ca/catalogo/a09002970-datos-meteorologicos-de-la-xema
https://analisi.transparenciacatalunya.cat/Medi-Ambient/Dades-meteorol-giques-di-ries-de-la-XEMA/7bvh-jvq2/about_data

API METEOCAT: https://apidocs.meteocat.gencat.cat/


dades temps real per fer la predicció:
https://analisi.transparenciacatalunya.cat/Transport/Incid-ncies-vi-ries-a-les-carreteres-de-Catalunya/5wp5-7t2p/about_data (NO TEMPS REAL, CADA X TEMPS)

https://analisi.transparenciacatalunya.cat/Transport/Incid-ncies-vi-ries-en-temps-real-a-Catalunya/uyam-bs37/about_data