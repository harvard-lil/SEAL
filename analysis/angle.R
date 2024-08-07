# Note: you may need to install the packages on your end
library(plm)
library(lmtest)
library(zoo)
library(car)

## Reward Shift Analysis
data <- read.csv("ivan-rlhf-analysis/output/2024-07-10-angles-old-new.csv") 
data <- data[, !names(data) %in% c("X")]

model <- lm(angles_uv ~ . - row_id  - align_shift - assistant_is_anthropomorphic_x - assistant_is_coherent_x - assistant_is_anthropomorphic_y - assistant_is_coherent_y - cos_v - cos_u, data = data)
print(vif(model))
coeftest_output <- coeftest(model)
coeftest_output[,'Pr(>|t|)'] <- p.adjust(coeftest_output[,'Pr(>|t|)'] , method = "bonferroni")
print(coeftest_output)


## Reward Analysis
# Note: the covariate that are removed follow from the VIF analysis
data <- read.csv("output/2024-07-10-linear-models-results.csv") 
data <- data[, !names(data) %in% c("X")]
model <- lm((reward-reward_n1) ~ . - topics - preference_rejected - assistant_is_coherent - assistant_is_anthropomorphic	, data = data)
print(vif(model))
coeftest_output <- coeftest(model, cluster = data$row_id)
coeftest_output[,'Pr(>|t|)'] <- p.adjust(coeftest_output[,'Pr(>|t|)'] , method = "bonferroni")
print(coeftest_output)


model <- lm(reward ~ . - topics - preference_rejected - assistant_is_coherent - assistant_is_anthropomorphic - reward_n1, data = data)
print(vif(model))

coeftest_output <- coeftest(model, cluster = data$row_id)
coeftest_output[,'Pr(>|t|)'] <- p.adjust(coeftest_output[,'Pr(>|t|)'] , method = "bonferroni")
print(coeftest_output)

model <- lm(reward_n1 ~ . - topics - preference_rejected - assistant_is_coherent - assistant_is_anthropomorphic - reward, data = data)
print(vif(model))

coeftest_output <- coeftest(model, cluster = data$row_id)
coeftest_output[,'Pr(>|t|)'] <- p.adjust(coeftest_output[,'Pr(>|t|)'] , method = "bonferroni")
print(coeftest_output)