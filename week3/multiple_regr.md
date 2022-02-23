1) what you found in your multiple regression analysis. Discuss the results for the associations between all of your explanatory variables and your response variable. Make sure to include statistical results (Beta coefficients and p-values) in your summary. 2) Report whether your results supported your hypothesis for the association between your primary explanatory variable and the response variable. 3) Discuss whether there was evidence of confounding for the association between your primary explanatory and response variable (Hint: adding additional explanatory variables to your model one at a time will make it easier to identify which of the variables are confounding variables); and 4) generate the following regression diagnostic plots:

a) q-q plot

b)  standardized residuals for all observations

c) leverage plot

d) Write a few sentences describing what these plots tell you about your regression model in terms of the distribution of the residuals, model fit, influential observations, and outliers. 

What to Submit: Submit the URL for your blog entry. The blog entry should include 1) the summary of your results that addresses parts 1-4 of the assignment, 2) the output from your multiple regression model, and 3) the regression diagnostic plots.



example 

After adjusting for potential confounding factors (list them), major depression (Beta=1.34, p=.0001) was significantly and positively associated with number of nicotine dependence symptoms. Age was also significantly associated with nicotine dependence symptoms, such that older participants reported a greater number of nicotine dependence symptoms (Beta= 0.76, p=.025).  


```
# regression analysis
reg1 = smf.ols('lnw ~ bmi_centered', data=data).fit()
print(reg1.summary())
```
<img width="770" alt="image" src="https://user-images.githubusercontent.com/22098104/155419768-110b24a1-8d2b-483f-b0c7-a1dca2aec87e.png">

```
# regression analysis with adding variables: age
reg1 = smf.ols('lnw ~ bmi_centered + age', data=data).fit()
print(reg1.summary())
```

<img width="779" alt="image" src="https://user-images.githubusercontent.com/22098104/155419937-db574dca-d51d-437c-91bb-a26c5fae6887.png">



```
# regression analysis with adding variables: race
reg1 = smf.ols('lnw ~ bmi_centered + age + white', data=data).fit()
print(reg1.summary())
```

<img width="799" alt="image" src="https://user-images.githubusercontent.com/22098104/155419974-90a8d00e-299f-43ed-bd81-39a1061a6c47.png">


```
# regression analysis with adding variables: age_cl
reg1 = smf.ols('lnw ~ bmi_centered + age + white + age_cl', data=data).fit()
print(reg1.summary())
```


<img width="780" alt="image" src="https://user-images.githubusercontent.com/22098104/155420041-6da5a7d9-94b3-4aa5-b02d-f47abd6e6538.png">


with each new variable R-squared shows that the percentage explained by the model increases.
