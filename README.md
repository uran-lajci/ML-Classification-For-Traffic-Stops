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

![image](https://user-images.githubusercontent.com/58117020/233866827-283991b7-adaa-4e17-bf82-8116f9abcf68.png)


## Faza 3

#### Ne fazen e 3 kemi zgjedhur edhe datasete te ndryshme per te bere testimet e algoritmeve edhe me te dhena te tjera, per te pare rezultatet dhe krahasuar me datasetin e pare, natyrisht se edhe keta datasete do te preprocesohen para se te implementohen algoritmet.

#### Gjithashtu tek algoritmi Naive Bayes kemi perdorur edhe laplance correction keshtu duke arritur nje rezultat shume me te mire ne krahasim me rezultatet e fazes se dyte.

#### Kemi kontrolluar gjithashtu nese kemi underfitting apo overfitting ne te dhenat tona, keshtu duke shikuar saktesine e te dhenave testuese dhe atyre trajnuese.

### Ne vazhdim mund te shohim rezultatet qe kemi arritur te ekzekutojme:
Keto tabela tregojn performancën e modeleve të ndryshme të ML në tre grupe të dhënash: NC Durham, NC Winston dhe një grup të dhënash fillestare. Çdo grup të dhënash po përdoret për të parashikuar driver_gender.

Algoritmet e ML të testuara përfshijnë Regresionin Logjistik (LR), Klasifikuesin e Pemës së Vendimit (DTC), Klasifikuesin e Rastit të Pyjeve (RFC), Naive Bayes (NB) dhe K-Fqinjët më të afërt (KNN). Për secilin algoritëm, u testuan tre ndarje të të dhënave: 0.2, 0.3 dhe 0.35. Këto ndarje ka të ngjarë të përfaqësojnë raportin e grupit të të dhënave që është përdorur për testim (me pjesën e mbetur të përdorur për trajnim).

Le të shohim disa gjetje specifike nga tabela:
* LR, Klasifikuesi i Pemës së Vendimit (DTC) dhe Klasifikuesi i Pyjeve të Rastit (RFC) tregojnë performancë të ngjashme në çdo grup të dhënash me një ndryshim të vogël midis ndarjeve të të dhënave. Performanca është më e lartë në grupin e të dhënave fillestare, më e ulët në NC Durham dhe më e ulët në NC Winston.
* NB në përgjithësi performoi më keq se algoritmet e tjera në të tre grupet e të dhënave. Kjo mund të jetë për shkak të karakteristikave specifike të të dhënave që nuk përputhen mirë me supozimet e Naive Bayes, ose për shkak të nevojës për akordim hiperparametër.
* KNN tregon më shumë variacione midis grupeve të të dhënave sesa algoritmet e tjera. Ai performon më keq në NC Winston sesa në dy grupet e tjera të të dhënave, por performanca në grupin e të dhënave fillestare rritet ndjeshëm ndërsa madhësia e ndarjes së testit rritet, duke arritur kulmin e saj në ndarjen 0.35.

![image](https://github.com/uran-lajci/Policing-ML-Model/assets/117693854/a99db8ce-c0cb-4432-a0c3-879aa428ee93)

![image](https://github.com/uran-lajci/Policing-ML-Model/assets/117693854/31871d92-fcf8-43be-b1ec-86db76c83957)

![image](https://github.com/uran-lajci/Policing-ML-Model/assets/117693854/0c5ced7f-98d4-4b1f-a0eb-7a00c75be602)

LR, DTC, RFC dhe NB po tregojnë saktësi të përsosur prej 100% në të gjitha grupet e të dhënave dhe me të gjitha ndarjet. Kjo është mjaft e pazakontë dhe mund të tregojë disa gjëra: detyra mund të jetë shumë e thjeshtë dhe lehtësisht e parashikueshme, mund të ketë një rrjedhje të të dhënave nga grupi i trajnimit në grupin e testimit, ose modelet mund të jenë tepër të përshtatura në të dhënat e trajnimit.

KNN tregon saktësi pak më të ulët se algoritmet e tjera, duke filluar nga 99,75% në 99,88% në varësi të grupit të të dhënave dhe ndarjes. Pavarësisht se janë më të ulëta se 100%, këto janë ende rezultate saktësie shumë të larta dhe në përgjithësi do të konsideroheshin të shkëlqyera në një skenar të botës reale.

![image](https://github.com/uran-lajci/Policing-ML-Model/assets/117693854/361045ad-501a-4fbc-9a46-a00000b04b70)

![image](https://github.com/uran-lajci/Policing-ML-Model/assets/117693854/571f45ed-8267-43e7-ace0-9decd91961a1)

![image](https://github.com/uran-lajci/Policing-ML-Model/assets/117693854/4e4638f7-fcf1-44eb-ace6-c8efa57c9117)

![image](https://github.com/uran-lajci/Policing-ML-Model/assets/117693854/499c70d1-846b-4dbb-be98-29a7a26d3e9f)

![image](https://github.com/uran-lajci/Policing-ML-Model/assets/117693854/1dd80085-07b1-47ff-a97a-9446a5153741)


