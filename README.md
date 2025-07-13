## Logistic Regression and Odds Ratio
Logistic regression and odds ratio using the example of COVID19 data

## Biref Overview:
 This project examines the determinants of COVID-19 vaccination in Germany using microdata from the 2023 European Social Survey (ESS11). The primary goal is to identify which individual characteristics, such as: age, income, education, internet usage, and institutional trust predict the likelihood of having received at least one dose of a COVID-19 vaccine. Italy is included as a comparative case to assess cross-national differences in these predictors. The analysis is based on a binary outcome variable derived from the ESS11 item \texttt{vacc19}. Logistic regression is employed to estimate the effects of relevant socio-demographic and attitudinal variables. The modeling approach includes data cleaning, dummy coding of categorical variables, and model reduction based on statistical significance. Parameters are estimated via maximum likelihood using the Newton-Raphson algorithm. Odds ratios are computed to facilitate interpretation. Final results for Germany show that older age, higher household income, and greater trust in the legal system and politicians are positively associated with the likelihood of COVID-19 vaccination. Conversely, being female and living in rural areas significantly decrease the odds of being vaccinated. In Italy, vaccination uptake is most strongly influenced by prior COVID-19 infection and larger household size. Trust in the police increases the odds of vaccination, whereas higher trust in political parties and higher levels of education are negatively associated. Living in less urban areas also corresponds to reduced vaccination likelihood. These results highlight differing national patterns: while institutional trust and socioeconomic status are key in Germany, personal experience and social context play a stronger role in Italy.

 ## Branch to Use:
 Main

## Installation
- Install the packages with your python3-pip packages
- Run the following command in the terminal:
```
pip install -r requirements.txt
```

## File Whereabouts
- Analysis for Germany: proejct.ipynb
- Analysis for Italy: project_IT.ipynb
- Analysis for France: project_FR.ipynb
- Analysis for Spain: project_ES.ipynb

## Workflow
Each file can act independently. The folder *data* should not be altered.

## Have Questions? Then Contact
- Md Mahmudul Haque (University e-mail: mahmudul.haque@tu-drotmund.de)
- Mehedi Rahman (University e-mail: mehedi.rahman@tu-dortmund.de)
- Sofiul Azam Sony (University e-mail: azam.sony@tu-dortmund.de)
- Nazmul Hasan Tanmoy (University e-mail: nazmul.tanmoy@tu-dortmund.de)