# Выполнил студент группы М8О-401Б-21 Знай Артемий

В файлах `lab1.ipynb` - `lab5.ipynb` лежат выполненные лабораторные работы 1-5

## Подведение итогов

### 1. K-Nearest Neighbors (KNN)

#### Регрессия:

| Модель                                      |       MSE |    MAE | $R^2$ |
| :------------------------------------------ | --------: | -----: | ----: |
| Sklearn (до улучшения)                      | 301414.49 | 349.37 |  0.38 |
| Sklearn (после улучшения)                   | 101359.24 | 207.16 |  0.79 |
| Собственная имплементация (до улучшения)    | 301308.79 | 349.13 |  0.38 |
| Собственная имплементация (после улучшения) | 101052.34 | 206.55 |  0.79 |

#### Классификация:

| Модель                                      | Accuracy | Precision | Recall | F1-scor |
| :------------------------------------------ | -------: | --------: | -----: | ------: |
| Sklearn (до улучшения)                      |   80.56% |    78.99% | 80.56% |  78.42% |
| Sklearn (после улучшения)                   |   83.36% |    82.46% | 83.36% |  82.55% |
| Собственная имплементация (до улучшения)    |   77.20% |    75.07% | 78.32% |  75.56% |
| Собственная имплементация (после улучшения) |   78.32% |    75.87% | 79.97% |  75.32% |

### 2. Linear Model

#### Регрессия:

| Модель                                      |       MSE |    MAE | $R^2$ |
| :------------------------------------------ | --------: | -----: | ----: |
| Sklearn (до улучшения)                      | 130868.65 | 264.34 |  0.73 |
| Sklearn (после улучшения)                   |  65975.97 | 185.85 |  0.86 |
| Собственная имплементация (до улучшения)    | 130868.65 | 264.34 |  0.73 |
| Собственная имплементация (после улучшения) |  85822.38 | 199.01 |  0.82 |

#### Классификация:

| Модель                                      | Accuracy | Precision | Recall | F1-score |
| :------------------------------------------ | -------: | --------: | -----: | -------: |
| Sklearn (до улучшения)                      |   85.23% |    84.58% | 85.23% |   84.33% |
| Sklearn (после улучшения)                   |   86.17% |    85.68% | 86.17% |   85.29% |
| Собственная имплементация (до улучшения)    |   81.68% |    83.42% | 81.68% |   82.28% |
| Собственная имплементация (после улучшения) |   85.05% |    84.35% | 85.05% |   84.22% |

### 3. Decision Tree

#### Регрессия:

| Модель                                      |       MSE |    MAE | $R^2$ |
| :------------------------------------------ | --------: | -----: | ----: |
| Sklearn (до улучшения)                      | 145584.74 | 247.89 |  0.70 |
| Sklearn (после улучшения)                   | 118536.23 | 228.46 |  0.76 |
| Собственная имплементация (до улучшения)    | 172541.46 | 275.09 |  0.64 |
| Собственная имплементация (после улучшения) | 120889.44 | 235.83 |  0.75 |

#### Классификация:

| Модель                                      | Accuracy | Precision | Recall | F1-score |
| :------------------------------------------ | -------: | --------: | -----: | -------: |
| Sklearn (до улучшения)                      |   82.43% |    82.17% | 82.43% |   82.29% |
| Sklearn (после улучшения)                   |   87.10% |    86.98% | 87.10% |   86.09% |
| Собственная имплементация (до улучшения)    |   84.11% |    83.71% | 83.11% |   83.87% |
| Собственная имплементация (после улучшения) |   87.48% |    87.05% | 87.48% |   87.08% |

### 4. Random Forest

#### Регрессия:

| Модель                                      |       MSE |    MAE | $R^2$ |
| :------------------------------------------ | --------: | -----: | ----: |
| Sklearn (до улучшения)                      | 153734.93 | 294.86 |  0.68 |
| Sklearn (после улучшения)                   | 118536.23 | 228.46 |  0.76 |
| Собственная имплементация (до улучшения)    | 155908.88 | 297.55 |  0.68 |
| Собственная имплементация (после улучшения) | 121329.04 | 231.17 |  0.75 |

#### Классификация:

| Модель                                      | Accuracy | Precision | Recall | F1-score |
| :------------------------------------------ | -------: | --------: | -----: | -------: |
| Sklearn (до улучшения)                      |   81.50% |    83.44% | 81.50% |   77.30% |
| Sklearn (после улучшения)                   |   89.16% |    89.09% | 89.16% |   88.52% |
| Собственная имплементация (до улучшения)    |   88.04% |    88.39% | 88.04% |   86.96% |
| Собственная имплементация (после улучшения) |   88.60% |    88.54% | 88.60% |   87.85% |

### 5. Gradient Boosting

#### Регрессия:

| Модель                                      |       MSE |    MAE | $R^2$ |
| :------------------------------------------ | --------: | -----: | ----: |
| Sklearn (до улучшения)                      | 126493.13 | 219.29 |  0.74 |
| Sklearn (после улучшения)                   |  63201.39 | 193.60 |  0.87 |
| Собственная имплементация (до улучшения)    | 122053.65 | 228.68 |  0.75 |
| Собственная имплементация (после улучшения) |  62979.43 | 193.22 |  0.87 |

#### Классификация:

| Модель                                      | Accuracy | Precision | Recall | F1-score |
| :------------------------------------------ | -------: | --------: | -----: | -------: |
| Sklearn (до улучшения)                      |   87.66% |    87.44% | 87.66% |   87.53% |
| Sklearn (после улучшения)                   |   89.53% |    89.32% | 89.53% |   89.08% |
| Собственная имплементация (до улучшения)    |   85.05% |    84.83% | 85.05% |   84.93% |
| Собственная имплементация (после улучшения) |   87.48% |    88.18% | 87.48% |   86.13% |

## Выводы

1. **K-Nearest Neighbors (KNN):**

   - Усовершенствование моделей привело к значительному снижению метрик ошибки как в регрессии, так и в классификации. Например, MSE в регрессии снизилась с 301414.49 до 101359.24 для реализации Sklearn.
   - Тем не менее, собственная имплементация KNN показала немного худшие результаты по сравнению с библиотечным вариантом, особенно в классификации.

2. **Linear Model:**

   - Линейные модели после улучшений демонстрируют хорошие результаты, особенно в регрессии. MSE для Sklearn модели сократилась почти вдвое — с 130868.65 до 65975.97.
   - Классификация с использованием линейных моделей также улучшилась, но разница между библиотечными и собственными реализациями меньше, чем у KNN.

3. **Decision Tree:**

   - Улучшение деревьев решений дало ощутимое увеличение точности классификации (Accuracy выросла с 82.43% до 87.10% для Sklearn) и уменьшение ошибки в регрессии.
   - Собственная имплементация после оптимизации практически догнала библиотечную по метрикам.

4. **Random Forest:**

   - Random Forest продемонстрировал значительное повышение точности классификации (Accuracy выросла с 81.50% до 89.16% для Sklearn) и улучшение метрик регрессии после оптимизации.
   - Собственная реализация немного уступает библиотечным моделям по всем метрикам, но разница невелика.

5. **Gradient Boosting:**
   - Градиентный бустинг показал наилучшие результаты среди всех методов, особенно в регрессии. Например, MSE снизилась до 63201.39 для Sklearn, что является лучшим значением среди всех моделей.
   - Классификация также продемонстрировала высокую точность и F1-score после улучшений.

---

### Общие выводы:

- Улучшение моделей дало значительное повышение качества во всех методах. Это демонстрирует важность корректной настройки гиперпараметров и оптимизации алгоритмов.
- Среди рассмотренных моделей **Gradient Boosting** оказался наиболее эффективным как для регрессии, так и для классификации, что делает его предпочтительным выбором для задач с высокими требованиями к точности.
- Собственные реализации показали сопоставимые результаты с библиотечными, что подтверждает корректность их работы. Однако использование библиотек, таких как Sklearn, позволяет добиться немного лучших результатов благодаря более оптимизированным алгоритмам.