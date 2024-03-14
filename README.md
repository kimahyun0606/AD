**Description**
머신런닝은 학습할 자료(데이터) 처리에 인간이 선 개입해 컴퓨터가 인식할 수 있게 한 후 학습하게 해서 모델을 만들어내는 것이다. 

인공신경망(ANN, Artificial Neural Network)은 사람의 신경망의 원리와 구조를 모방하여 만든 기계학습 알고리즘즘

**데이터 상관관계 분석(전처리 과정)**

```
library(readxl)
library(ggplot2)
library(dplyr)
library(rlang)

AD <- read.csv("AD.csv")
AD <- AD[, -1]
head(AD)
install.packages("ltm")
library(ltm)
```

# correlations test 반복문
```
cor_results <- data.frame(Variable1 = character(), Variable2 = character(), Correlation = numeric(), P_Value = numeric(), stringsAsFactors = FALSE)
```

# for 루프로 상관관계 검정 수행 및 결과 데이터프레임에 추가
```
for (i in 2:30) { 
  cor_test <- cor.test(AD[, i], AD$AD)
```  
 
 # 결과를 데이터프레임에 추가
```
  cor_results <- rbind(cor_results, data.frame(
    Variable1 = "AD",
    Variable2 = names(AD)[i],
    Correlation = cor_test$estimate,
    P_Value = cor_test$p.value
  ))
}
```

# 결과 데이터프레임 출력
```
print(cor_results)
cor_results

write.csv(cor_results, file ="cor_results.csv")
```


# correlations test (MetRate제거)
```
AD_1 <- AD %>% select(-MetRate)
head(AD_1)

cor_results_1 <- data.frame(Variable1 = character(), Variable2 = character(), Correlation = numeric(), P_Value = numeric(), stringsAsFactors = FALSE)
```

# for 루프로 상관관계 검정 수행 및 결과 데이터프레임에 추가
```
for (i in 2:29) { 
  cor_test <- cor.test(AD_1[, i], AD_1$AD)
```

  # 결과를 데이터프레임에 추가
```
  cor_results_1 <- rbind(cor_results_1, data.frame(
    Variable1 = "AD",
    Variable2 = names(AD_1)[i],
    Correlation = cor_test$estimate,
    P_Value = cor_test$p.value
  ))
}
```

# 결과 데이터프레임 출력
```
print(cor_results_1)
cor_results_1

write.csv(cor_results_1, file ="cor_results_1.csv")




t_test_results <- data.frame(Variable1 = character(), Variable2 = character(), statistic = numeric(), P_Value = numeric())
```

# t- test 반복문
```
AD_1 <- AD %>% filter(AD ==1)
summary(AD_1)

AD_0 <- AD %>% filter(AD==0)
summary(AD_0)

for (i in 2:30) { 
  t_test <- t.test(AD_1[, i], AD_0[, i])
```
  
# 결과를 데이터프레임에 추가
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

# 결과 데이터프레임 출력
```
print(cor_results)
cor_results

write.csv(cor_results, file ="cor_results.csv")
```


# correlations test (MetRate제거)
```
AD_1 <- AD %>% select(-MetRate)
head(AD_1)

cor_results_1 <- data.frame(Variable1 = character(), Variable2 = character(), Correlation = numeric(), P_Value = numeric(), stringsAsFactors = FALSE)
```

# for 루프로 상관관계 검정 수행 및 결과 데이터프레임에 추가
```
for (i in 2:29) { 
  cor_test <- cor.test(AD_1[, i], AD_1$AD)
```
  
  # 결과를 데이터프레임에 추가
```
  cor_results_1 <- rbind(cor_results_1, data.frame(
    Variable1 = "AD",
    Variable2 = names(AD_1)[i],
    Correlation = cor_test$estimate,
    P_Value = cor_test$p.value
  ))
}
```

# 결과 데이터프레임 출력
```
print(cor_results_1)
cor_results_1

write.csv(cor_results_1, file ="cor_results_1.csv")




t_test_results <- data.frame(Variable1 = character(), Variable2 = character(), statistic = numeric(), P_Value = numeric())
```

# t- test 반복문
```
AD_1 <- AD %>% filter(AD ==1)
summary(AD_1)

AD_0 <- AD %>% filter(AD==0)
summary(AD_0)

for (i in 2:30) { 
  t_test <- t.test(AD_1[, i], AD_0[, i])
```
  
  # 결과를 데이터프레임에 추가
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

# 결과 데이터프레임 출력
```
print(cor_results)
cor_results

write.csv(cor_results, file ="cor_results.csv")
```

# correlations test (MetRate제거)
```
AD_1 <- AD %>% select(-MetRate)
head(AD_1)

cor_results_1 <- data.frame(Variable1 = character(), Variable2 = character(), Correlation = numeric(), P_Value = numeric(), stringsAsFactors = FALSE)
```

# for 루프로 상관관계 검정 수행 및 결과 데이터프레임에 추가
```
for (i in 2:29) { 
  cor_test <- cor.test(AD_1[, i], AD_1$AD)
```
  
  # 결과를 데이터프레임에 추가
```
  cor_results_1 <- rbind(cor_results_1, data.frame(
    Variable1 = "AD",
    Variable2 = names(AD_1)[i],
    Correlation = cor_test$estimate,
    P_Value = cor_test$p.value
  ))
}
```

# 결과 데이터프레임 출력
```
print(cor_results_1)
cor_results_1

write.csv(cor_results_1, file ="cor_results_1.csv")




t_test_results <- data.frame(Variable1 = character(), Variable2 = character(), statistic = numeric(), P_Value = numeric())
```

# t- test 반복문
```
AD_1 <- AD %>% filter(AD ==1)
summary(AD_1)

AD_0 <- AD %>% filter(AD==0)
summary(AD_0)

for (i in 2:30) { 
  t_test <- t.test(AD_1[, i], AD_0[, i])
```
  
  # 결과를 데이터프레임에 추가
```
  t_test_results <- rbind(t_test_results, data.frame(
    Variable1 = "AD",
    Variable2 = "non-AD",
    variable3 = names(AD_1)[i],
    t_value = t_test$statistic,
    P_Value = t_test$p.value
  ))
}

t_test_results
write.csv(t_test_results, file="t_test_results.csv")
```






AD_ANN 5개 Fe, Cu, Pb, Na_Mg

# 패키지 불러오기
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

# ADdata.csv불러오기
```
df = pd.read_csv("normalized_AD.csv")
df = df.iloc[:, 1:]
```
```
df = df[['AD','Fe','Cu','Pb','Na_Mg']]
```

#데이터 프레임 구성
```
df = pd.DataFrame(df)
df.head()
```
```
df.info()
```

# 데이터 개수 세기
```
print(df['AD'].value_counts())
print(df['sex'].value_counts())
```

# 칼럼 개수
```
print(df.columns)
print(df.shape[1])
```

# matplotlib 한글 폰트 추가하기
```
import matplotlib.font_manager as fm

font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')

f = [f.name for f in fm.fontManager.ttflist]
print(len(font_list))

plt.rcParams["font.family"] = 'Malgun Gothic'
```

# 음수 표시
```
plt.rcParams['axes.unicode_minus'] = False
```
```
X = df.loc[:,df.columns !='AD']
y = df['AD']
```
```
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
```

# 훈련데이터 평가 데이터 분리
```
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=5) 
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
```

# 그리드 서치로 하이퍼 파라미터 조정
# ANN 적용하기 sklearn을 활용
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
y_pred = ann.predict(x_test)
print(classification_report(y_test,y_pred))
```
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
```
print("roc_auc_score:",roc_auc_score(y_test,y_pred, multi_class='raise'))
```
