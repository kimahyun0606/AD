**Description**
머신런닝은 학습할 자료(데이터) 처리에 인간이 선 개입해 컴퓨터가 인식할 수 있게 한 후 학습하게 해서 모델을 만들어내는 것이다. 



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

