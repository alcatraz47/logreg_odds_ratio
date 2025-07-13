import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/ESS11.csv", sep=";", encoding="utf-8",
                usecols=["idno", "respc19a", "vacc19",	"cntry",
                        "eisced","hinctnta", "hhmmb", "netusoft",
                        "gndr", "maritalb", "domicil", "agea", "trstprl",
                        "trstlgl", "trstplc", "trstplt", "trstprt"])
# Show basic structure
print("Shape of the data:", df.shape)
print(df.isna().sum())

# GERMANY data
# df = df[df['cntry'] == 'DE']
# print("Length of the Germany data:", df.shape)

df = df[df['cntry'] == 'IT']
print("Length of the Italy data:", df.shape)
print(df.isna().sum())

# Recode vacc19
df = df[df['vacc19'].isin([1, 2])]  # keep only valid responses
df['vacc19_binary'] = df['vacc19'].map({1: 1, 2: 0})

df['gndr'] = df['gndr'].replace({1: 0, 2: 1})

# -----------------------------------------
# 1. Variables with missing codes {7, 8, 9}
# -----------------------------------------
vars_789 = ['netusoft', 'domicil']
for var in vars_789: 
    df = df[~df[var].isin([7,8,9])]

# -----------------------------------------
# 2. Variables with missing codes {77, 88, 99}
# -----------------------------------------
vars_778899 = ['maritalb', 'hinctnta', 'hhmmb',
            'trstprl', 'trstlgl', 'trstplc', 'trstplt', 'trstprt']
for var in vars_778899:
    df = df[~df[var].isin([77,88,99])]

# -----------------------------------------
# 3. Variables with missing codes {7, 8, 9, 77, 88, 99}
# (if any variable had all of these, we could use this, but none does here)
# so we skip this group for now

# -----------------------------------------
# 4. Variables with specific missing codes
# -----------------------------------------

# gender (only 9 means missing)
df = df[df['gndr'] != 9]

# education level (more specific)
df = df[~df['eisced'].isin([55,77,88,99])]

# age (999 means missing)
df = df[df['agea'] != 999]

# country does not need cleaning (mandatory)

# -----------------------------------------
# show cleaned data
# -----------------------------------------
print(df)
print(df.isnull().sum())

df['hinctnta'] = df['hinctnta'].fillna(df['hinctnta'].median())
df = df.dropna()
print(df.isnull().sum())

# saving for the country GERMANY
# df.to_csv("data/ESS11_de_cleaned.csv", index=False, sep=";")
# saving for the country ITALY
# df.to_csv("data/ESS11_IT_cleaned.csv", index=False, sep=";")

ctab = None
# check cross-tabulations for each categorical variable before fitting:
for var in ['gndr', 'maritalb', 'domicil', 'cntry', 'eisced', "respc19a_binary",
            "trstprl", "trstlgl", "trstplc", "trstplt", "trstprt"]:
    ctab = pd.crosstab(df[var], df['vacc19_binary'])
    print(f"\nCross-tabulation for {var}:\n", ctab)
# get vaccination rate by income decile
vacc_rates = df.groupby('hinctnta')['vacc19_binary'].mean()

# plot
plt.plot(vacc_rates.index, vacc_rates.values, marker='o')
plt.xlabel('Household income decile')
plt.ylabel('Proportion vaccinated')
plt.title('Vaccination rate by income decile')
plt.show()

# get vaccination rate by income decile
vacc_rates = df.groupby('netusoft')['vacc19_binary'].mean()

# plot
plt.plot(vacc_rates.index, vacc_rates.values, marker='o')
plt.xlabel('Household internet usage frequency')
plt.ylabel('Proportion vaccinated')
plt.title('Vaccination rate by internet usage frequency')
plt.show()
# Assuming df is still loaded from previous cleaning steps


# Assuming df is still loaded from previous cleaning steps

# # Redefine the list of predictors for GERMANY
# predictors = [
#     'agea',
#     'respc19a_binary',
#     'hinctnta',
#     'hhmmb',
#     'netusoft',
#     'gndr',
#     'trstprl', 
#     'trstlgl', 
#     'trstplc', 
#     'trstplt'
# ]

# Redefine the list of predictors
predictors = [
    'agea',
    'respc19a_binary',
    'hinctnta',
    'hhmmb',
    'netusoft',
    'gndr',
    "trstlgl",
    "trstplc",
    "trstplt",
    "trstprt"
]
# Dummy code relevant categorical variables
df = pd.get_dummies(df, columns=['maritalb', 'domicil', 'eisced'], drop_first=True)

# Add all dummies to predictors
predictors += [col for col in df.columns if any(x in col for x in ['maritalb_', 'domicil_', 'eisced_'])]

print("Predictors after dummy coding:\n", predictors)
# Make sure there are no non-numeric columns:
for col in predictors:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# # Remove any rows with NA (if missing)
# df_model = df.dropna(subset=predictors + ['vacc19_binary'])
# Drop missing rows
df_model = df.dropna(subset=predictors + ['vacc19_binary'])
# Create design matrix
X = df_model[predictors]
X = sm.add_constant(X)

# Response variable
y = df_model['vacc19_binary']

# Confirm again that X is fully numeric
print(X.dtypes)

# Fit the model
logit_model = sm.Logit(y, X.astype(float))
result = logit_model.fit()
# Print the summary
print(result.summary())
# You can also get odds ratios:
odds_ratios = pd.Series(np.exp(result.params), name="Odds Ratio")
print("\nOdds Ratios:\n", odds_ratios)

# initialize design matrix
X = df[predictors]
X = sm.add_constant(X)

# backward selection loop
current_predictors = X.columns.tolist()

while True:
    model = sm.Logit(y, X[current_predictors].astype(float)).fit(disp=0)
    pvalues = model.pvalues.drop('const')
    max_p = pvalues.max()
    if max_p > 0.05:
        worst_predictor = pvalues.idxmax()
        print(f"Dropping {worst_predictor} with p-value {max_p:.4f}")
        current_predictors.remove(worst_predictor)
    else:
        break

# final model
final_model = sm.Logit(y, X[current_predictors].astype(float)).fit()
print(final_model.summary())

# odds ratios
odds_ratios = pd.Series(np.exp(final_model.params), name="Odds Ratio")
print("\nOdds Ratios:\n", odds_ratios)