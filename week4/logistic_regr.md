This week's assignment is to test a logistic regression model. 

1st part of the code

```
# -*- coding: utf-8 -*-

import numpy as numpy
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn 
import matplotlib.pyplot as plt

# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%.2f'%x)

data = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/adult_services.csv')
data.dropna(inplace=True)
data = data.drop(columns=["Unnamed: 0"]
```

i'll take as a response variable "unsafe", its a binary variable where 1 stands for unprotected sex with client of any kind. lets investigate the relationship with worker's race.

```
# logistic regression 
lreg1 = smf.logit(formula = 'unsafe ~ black', data = data).fit()
print(lreg1.summary())
# odds ratios
print ("Odds Ratios")
print (numpy.exp(lreg1.params))

# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print(numpy.exp(conf))

```

<img width="691" alt="image" src="https://user-images.githubusercontent.com/22098104/155791273-37a69da2-651c-46e9-bea0-2a123b3fd23e.png">


p value (0.037) is less than 0.05 so the relationship between these variables appears to be significant
odds ratio (0.58) is < 1, which means that the worker is less likely to participate in an unsafe sex if they are black.
95% CI = 0.34-0.97


now i'll add another variable: massage_cl 1 stands for Gave Client a Massage

```
lreg2 = smf.logit(formula = 'unsafe ~ black + massage_cl', data = data).fit()
print(lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print(numpy.exp(conf))
```

<img width="682" alt="image" src="https://user-images.githubusercontent.com/22098104/155795080-0a65f1b6-4aac-43d6-b4ab-0e3bd3e2a526.png">

the results suggest that if worker gave a massage it's likely that the sex will be safe (OR= 0.55, 95% CI=0.45-0.68, p=.0.0001)


adjusting for other potential confounding factors

1) customer being a regular
```
lreg2 = smf.logit(formula = 'unsafe ~ black + massage_cl + reg', data = data).fit()
print(lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print(numpy.exp(conf))
```

<img width="690" alt="image" src="https://user-images.githubusercontent.com/22098104/155797795-4cd6e9ea-431c-437c-98f1-83c01b34ce26.png">

no statistical significance

2) there was a second worker during the session

provider_second

```
lreg2 = smf.logit(formula = 'unsafe ~ black + massage_cl + reg + provider_second', data = data).fit()
print(lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print(numpy.exp(conf))
```

<img width="723" alt="image" src="https://user-images.githubusercontent.com/22098104/155798444-2bf3020e-52c0-444f-abef-36a55ba99f6c.png">

no statistical significance

3) race of the client

```
lreg2 = smf.logit(formula = 'unsafe ~ black + massage_cl + reg + provider_second + black_cl', data = data).fit()
print(lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print(numpy.exp(conf))
```

<img width="734" alt="image" src="https://user-images.githubusercontent.com/22098104/155798812-79df54ab-6a1c-4fe4-a2e1-68afb1767b48.png">

no statistical significance

## To sum up, after adjusting for potential confounding factors (race of the client, second worker during the session, customer being a regular), the odds of having unsafe sex were 0.58 if sex worker's race is black (OR= 0.58, 95% CI= 0.34-0.97, p=0.037). Giving a massage was also significantly associated with having safe sex, such that if the massage was given people were significantly less likely to participate in unsafe sexual practices.
