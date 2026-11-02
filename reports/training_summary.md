# Resumen del entrenamiento

- Modelo seleccionado: LogisticRegression
- Variable objetivo: status
- Variables predictoras: x, y, speed, battery, motor_temp, vibration, motor_current, obstacle_distance
- Método de evaluación: GroupShuffleSplit por fleet_id
- Flotas de prueba: F001, F002, F009, F016, F018
- F1 macro: 0.9368
- Accuracy: 0.9333
- Precision macro: 0.9362
- Recall macro: 0.9373

## Comparación de modelos

| model              |   f1_macro |   accuracy |   precision_macro |   recall_macro |   train_samples |   test_samples |
|:-------------------|-----------:|-----------:|------------------:|---------------:|----------------:|---------------:|
| LogisticRegression |   0.936752 |   0.933333 |          0.936244 |       0.937288 |            2250 |            750 |
| RandomForest       |   0.934975 |   0.932    |          0.935269 |       0.934773 |            2250 |            750 |

## Reporte de clasificación

```text
              precision    recall  f1-score   support

    CRITICAL       0.96      0.96      0.96       167
      NORMAL       0.94      0.94      0.94       308
     WARNING       0.91      0.91      0.91       275

    accuracy                           0.93       750
   macro avg       0.94      0.94      0.94       750
weighted avg       0.93      0.93      0.93       750

```

La matriz de confusión está disponible en `reports/confusion_matrix.csv`.