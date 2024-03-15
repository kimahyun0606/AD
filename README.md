**Description**
머신러닝은 학습할 자료(데이터) 처리에 인간이 선 개입해 컴퓨터가 인식할 수 있게 한 후 학습하게 해서 모델을 만들어내는 것이다. 

머신러닝은
1. 데이터 수집
2. 데이터 전처리
3. 분석 모델링을 통해 모델 selection
4. 모델 fitting, training(split)
5. 모델 평가 검증


# 데이터 수집
pd.read_csv로 AD data를 불러온다. 
```
df = pd.read_csv("normalized_AD.csv)
df = df.iloc[:, 1:]
```

![image](https://github.com/kimahyun0606/AD/assets/162280996/dcb82998-7a91-4cc8-beb0-c09d2569a8bc)


# 데이터 프레임 구성
df = pd.DataFrame(df)
df.head()
![image](https://github.com/kimahyun0606/AD/assets/162280996/687303db-7cd1-48d6-a881-8eac672cfc56)
(이후 피처 특정하면)
df = df[['AD','Fe','Cu','Pb','Na_Mg']]
df.head()
![image](https://github.com/kimahyun0606/AD/assets/162280996/62d9515f-fb36-44d3-94cd-785fc36f194b)

**데이터 상관관계 분석(전처리 과정)**

전처리과정은 데이터가 목적에 맞게 최적화되어 있지 않기 때문에 수집데이터를 그대로 사용할 경우 잘못된 분석결과를 도출하거나 분석의 성능이 떨어질 수 있다. 데이터 전처리 과정은 매우 중요하게 다루어 지고 있다. 


> 이 과정을 통해 값이 누락된 데이터 결측치와 일반적인 범위에서 벗어난 값 이상치를 제외하거나 적절히 수정하여 분석의 정확성을 높인다. 
> 변수들 간의 영향력을 조정하기 위해 정규화와 표준화를 사용한다. 데이터 변수들 간의 범위가 다를 경우 분석의 성능이 하락할 수 있기 때문이다. 
> 전체 데이터 중 분석 영향력이 떨어지는 변수를 제거하여 분석의 성능을 높이는 전처리 과정인 '피처 선택'과 수집 데이터에 존재하는 변수들 간의 연산을 통해 파생 변수를 생성하는 것을 '피처 엔지니어링'과성을 통해 모델의 복잡성을 줄이고 효율성을 높이고, 모델의 예측 성능을 향상시킬수 있다.

library(readxl)
library(ggplot2)
library(dplyr)
library(rlang)

AD <- read.csv("AD.csv")
AD <- AD[, -1]
head(AD)
install.packages("ltm")
library(ltm)


### correlations test 반복문
```
cor_results <- data.frame(Variable1 = character(), Variable2 = character(), Correlation = numeric(), P_Value = numeric(), stringsAsFactors = FALSE)
```

> ### for 루프로 상관관계 검정 수행 및 결과 데이터프레임에 추가
```
for (i in 2:30) { 
  cor_test <- cor.test(AD[, i], AD$AD)
```   
>> ### 결과를 데이터프레임에 추가
```
  cor_results <- rbind(cor_results, data.frame(
    Variable1 = "AD",
    Variable2 = names(AD)[i],
    Correlation = cor_test$estimate,
    P_Value = cor_test$p.value
  ))
 }
```

### 결과 데이터프레임 출력
```
print(cor_results)
cor_results

write.csv(cor_results, file ="cor_results.csv")
```

=========


### correlations test (MetRate제거)
```
AD_1 <- AD %>% select(-MetRate)
head(AD_1)

cor_results_1 <- data.frame(Variable1 = character(), Variable2 = character(), Correlation = numeric(), P_Value = numeric(), stringsAsFactors = FALSE)
```

> ### for 루프로 상관관계 검정 수행 및 결과 데이터프레임에 추가
```
for (i in 2:29) { 
  cor_test <- cor.test(AD_1[, i], AD_1$AD)
```

>> ### 결과를 데이터프레임에 추가
```
  cor_results_1 <- rbind(cor_results_1, data.frame(
    Variable1 = "AD",
    Variable2 = names(AD_1)[i],
    Correlation = cor_test$estimate,
    P_Value = cor_test$p.value
  ))
}
```

### 결과 데이터프레임 출력
```
print(cor_results_1)
cor_results_1

write.csv(cor_results_1, file ="cor_results_1.csv")
```
=======


t_test_results <- data.frame(Variable1 = character(), Variable2 = character(), statistic = numeric(), P_Value = numeric())

### t- test 반복문
```
AD_1 <- AD %>% filter(AD ==1)
summary(AD_1)

AD_0 <- AD %>% filter(AD==0)
summary(AD_0)
```
```
> for (i in 2:30) { 
  t_test <- t.test(AD_1[, i], AD_0[, i])
```
  
>> ### 결과를 데이터프레임에 추가
```
  t_test_results <- rbind(t_test_results, data.frame(
    Variable1 = "AD",
    Variable2 = "non-AD",
    variable3 = names(AD_1)[i],
    t_value = t_test$statistic,
    P_Value = t_test$p.value
  ))
}
```


```
t_test_results
write.csv(t_test_results, file="t_test_results.csv")
```



# 분석 모델링 선택
분석모델링은 회귀 분석 모델링(column 묶기 > 원핫/라벨인코더 > train_test_split > Stnadard/MinMaxscaler > 분석)
분류 분석 모델링(column 묶기 > 원핫/라벨인코더 > train_test_split > Standard/MinMaxScaler > 분)
![image](https://github.com/kimahyun0606/AD/assets/162280996/d04f82b2-f314-4a02-b063-bdd72278a108)
5개 모델을 생성하였다. 
5개 데이터 처리
from sklearn.model_selection import GridSearchCV
ann + MLPClassifier


AD_ANN 5개 Fe, Cu, Pb, Na_Mg

### 패키지 불러오기
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

### ADdata.csv불러오기
```
df = pd.read_csv("normalized_AD.csv")
df = df.iloc[:, 1:]
```
```
df = df[['AD','Fe','Cu','Pb','Na_Mg']]
```

### 데이터 프레임 구성
```
df = pd.DataFrame(df)
df.head()
```
```
df.info()
```

### 데이터 개수 세기
```
print(df['AD'].value_counts())
print(df['sex'].value_counts())
```

### 칼럼 개수
```
print(df.columns)
print(df.shape[1])
```

### matplotlib 한글 폰트 추가하기
```
import matplotlib.font_manager as fm

font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')

f = [f.name for f in fm.fontManager.ttflist]
print(len(font_list))

plt.rcParams["font.family"] = 'Malgun Gothic'
```

### 음수 표시
```
plt.rcParams['axes.unicode_minus'] = False
```
```
X = df.loc[:,df.columns !='AD']
y = df['AD']
```

여기까지는 5개 모델링 코드가 같다.  

### 데이터 split 분리
#라이브러리 불러오기

ANN
```
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
```

LR
```
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import statsmodels.api as sm
```

NB
```
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
```

RF
```
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
라이브러리 from sklearn.ensemble import RandomForestClassifier
```

SVM
```
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
```

#훈련데이터 평가 데이터 분리
> X, y로 일단 split 하고, train_test_split(X, y, stratify, test_size, random_state)
```
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=5) 
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
```

#데이터스케일링?

# 분석 모델 적용
> 
### 그리드 서치로 하이퍼 파라미터 조정(검증)
### ANN 적용하기 sklearn을 활용
```
ann = MLPClassifier()

params = {
    'hidden_layer_sizes':[10,30,50,100],
    'activation':['relu','tanh'],
    'solver':['adam','sgd'],
    
}

grid = GridSearchCV(ann,param_grid=params,verbose=1)
grid.fit(x_train,y_train)

print(grid.best_params_)
```

```
ann = MLPClassifier(hidden_layer_sizes= 50,solver='sgd',activation='tanh')
ann.fit(x_train,y_train)
**y_pred = ann.predict(x_test)**
print(classification_report(y_test,y_pred))
```

# 데이터 분석 결과 검증

```
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
labels = [0,1]
y_test =label_binarize(y_test,classes=labels)
y_pred =label_binarize(y_pred,classes=labels)
```

```
n_classes = 1
fpr = dict()
tpr =dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(y_test[:,i],y_pred[:,i])
    roc_auc[i] =auc(fpr[i],tpr[i])
```

```
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
for idx, i in enumerate(range(n_classes)):
    plt.subplot(141+idx)
    plt.plot(fpr[i],tpr[i],label='ROC curve (area = %0.2f)'%roc_auc[i])
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Postive Rate')
    plt.legend(loc="lower right")
plt.show()
```
결과 : ![image](https://github.com/kimahyun0606/AD/assets/162280996/e709dd36-87ad-4d2d-ae27-895836f7ed9d)

#'함수이름'.predict_proba('X test 데이터) : roc auc score를 구할때
```
print("roc_auc_score:",roc_auc_score(y_test,y_pred, multi_class='raise'))
```
결과 : roc_auc_score: 0.5666666666666667
