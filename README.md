# Policing-ML-Model

#### Per te ekzekutuar kete paze te projektit fillimisht duhet te instaloni librarine pandas dhe numpy duke shkruar ne terminal:
```python
pip install pandas
pip install numpy
```
#### dhe me pas te shkruani ne terminal komanden:
```python
python model_preparation.py
```

> Ne kete projekt ne kemi perdorur nje dataset te huazuar nga Kaggle ne linkun ne vijim: [Stanford Open Policing Project](https://www.kaggle.com/datasets/faressayah/stanford-open-policing-project).
I cili i permban keto kolona:
```jupiter
- stop_date
- stop_time
- county_name
- driver_gender
- driver_age_raw
- driver_age
- driver_race
- violation_raw
- violation
- search_conducted
- search_type
- stop_outcome
- is_arrested
- stop_duration
- drugs_related_stop
```
> Ne do te zhvillojme nje parashikim se: a do te arrestohet nje person a jo dhe cka do te jete rezultati i ndaleses ne baze te dhenave te tij.
>> Per arritjen e parashikimit te target vleres tone do te perdorim klasifikimin.

***Algoritmet e klasifikimit qe pretendojm qe ti perdorim qe te gjejme se cili na pershtatet dhe na rezulton me rezultate me te mira jane:***
```python
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Naive Bayes
5. K-Nearest Neighbors (KNN)
```



Gjitheashtu do te shohim mundesine se si do te performonin Neural Networks ne problemin tone.

***Ndersa per testim sigurisht se mbeshtetja krysore do te jete Corss-Validation , por gjithashtu mund te perdorim edhe metrika tjera si ROC Curve, Precision-Recal, Confusion Matrix dhe AUC***


## Faza 2.

#### Ne fazen e dyte kemi vazhduar me trajnimin e modelit qe kemi permendur ne fazen e pare te projektit tone, keshtu ku fillimisht kemi marrur datasetin te cilin e kemi perpunuar ne fazen e pare dhe kemi bere trajnimin e modelit ne menyra te ndryshme per disa target klasa te ndryshme, te cilat jane:

```text
- stop_outcome
- is_arrested
- driver_gender
- age_group
- violation
```

#### per ekzekutimin e projektit fillimisht ne terminal shkruajm

```jupiter
python .\model_training.py
```
#### me pas duhet te shenojme numrin se sa rekorde mendojme te marrim ne menyre qe te bejme trajnimin e modelit, me pas japim nje pergjigje me "y" nese deshirojme qe te kemi nje zgjedhje te rekordeve ne menyre te rendomte, dhe me pas zgjedhim atributin qe deshirojme qe te bejme predict: stop_outcome, driver_gender, age_group, is_arrested or violation. Gjithashtu ne fund fare zgjedhim edhe algoritmin me te cilin behet trajnimi i modelit, pas zgjedhjes se nje shembulli rezultatet mund te jene si ne vijim:

```console
PS C:\Users\Elvir Misini\Desktop\masterFK\S2\ML\Policing-ML-Model> python .\model_training.py
Write the number of rows you want to use from 86099 that are in the dataset: 80000
Write y if you want to randomly select rows: n
Choose which attribute you want to predict from these attributes: stop_outcome, driver_gender, age_group, is_arrested or violation
Attribute to be predicted: driver_gender
Choose the model to train (LR, DTC, RFC, NB, KNN): RFC
MODEL : 73.12
Confusion Matrix
[[    4  6428]
 [   22 17546]]
================
              precision    recall  f1-score   support

           F       0.15      0.00      0.00      6432
           M       0.73      1.00      0.84     17568

    accuracy                           0.73     24000
   macro avg       0.44      0.50      0.42     24000
weighted avg       0.58      0.73      0.62     24000

====================================================
====================================================
Accuracy: 0.73 (+/- 0.01)
```

#### P.S. Nese kemi zgjedhur is_arrested ose driver_gender, ne fund na paraqitet edhe vizualizimi i ROC Curve:

![image](https://user-images.githubusercontent.com/58117020/233801053-fb08ab7a-e48c-4e67-99d6-fdf99491554d.png)

# Testimet

![image](https://user-images.githubusercontent.com/117693854/233863104-35fc3665-e94d-4537-b81c-d3e2c7292a8c.png)

![image](https://user-images.githubusercontent.com/117693854/233863136-938a4b67-c76c-4147-a976-d7df1f1a7239.png)

![image](https://user-images.githubusercontent.com/117693854/233863153-286e88e9-9324-4645-8124-5c9c99ec3775.png)

![image](https://user-images.githubusercontent.com/117693854/233865871-5a3caa6d-afdd-40a8-b9dd-ecf38343fa5e.png)
