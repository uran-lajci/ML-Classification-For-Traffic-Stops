# Policing-ML-Model

> Ne kete projekt ne kemi perdorur nje dataset te huazuar nga Kaggle ne linkun ne vijim: [Stanford Open Policing Project](https://www.genome.gov/) .
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
> Ne do te zhvillojme nje parashikim se a do te arrestohet nje person os jo ne baze te te dhenave te tij.
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
