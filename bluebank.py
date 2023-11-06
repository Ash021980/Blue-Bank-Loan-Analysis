# -*- coding: utf-8 -*-

"""
Created on Mon Oct 30 16:54:29 2023

@author: Malcomb C. Brown

Blue Bank Loan Analysis
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")


# Load Blue Bank Loan Data
raw = pd.read_json("Data/loan_data_json.json")

# Save a copy of the raw data
loan_df = raw.copy()

# Inspect the dataset
# loan_df.head()
# loan_df.tail()
# loan_df.info()

# I want to clean the column labels by replacing the'.' with '_'
# Clean column labels
loan_df.columns = loan_df.columns.str.replace(".", "_")
# loan_df.info()

# Verify that there are no Null values
loan_df.isnull().sum()

# Check for duplicate records
# loan_df[loan_df.duplicated()]
loan_df.duplicated().sum()
# There are no duplicate records

# Let's look at the summary statistics
loan_df.describe(include="O")

# Change 'pupose' data type from object to category
loan_df["purpose"] = loan_df.purpose.astype("category")
# loan_df.info()

# What are the different values for 'purpose'?
loan_df.purpose.nunique()
loan_df.purpose.unique()

# How are the 'purpose' values distributed?
loan_df.purpose.value_counts()
# Show as a percent
loan_df.purpose.value_counts(normalize=True)

# Get summary statistics for the 'int_rate', 'dti', and 'fico' features
loan_df.int_rate.describe()
loan_df.fico.describe()
loan_df.dti.describe()

# Calculate the actual annual income
loan_df["annual_inc"] = np.exp(loan_df.log_annual_inc)
# loan_df.info()

# Visualize the distribution of 'fico' scores
loan_df.fico.plot.hist(title="FICO Score Distribution")
plt.show()

# Create a new feature based on the rating assigned for different ranges
# of the FICO credit scores.

# fico: The FICO credit score of the borrower.
# - 300 - 400: Very Poor
# - 401 - 600: Poor
# - 601 - 660: Fair
# - 661 - 780: Good
# - 781 - 850: Excellent

# Long inefficient code, works with a small dataset but will slow down
# performance the larger the dataset becomes.

# credit_ratings = []
# for rating in loan_df.fico:
#     if rating > 780:
#         credit_ratings.append("Excellent")
#     elif rating > 660:
#         credit_ratings.append("Good")
#     elif rating > 600:
#         credit_ratings.append("Fair")
#     elif rating > 400:
#         credit_ratings.append("Poor")
#     elif rating >= 300:
#         credit_ratings.append("Very Poor")
#     else:
#         credit_ratings.append("Unknown")

# loan_df["credit_rating"] = credit_ratings

# Better, vectorized, way to create the new feature 'credit_rating'.
mapper = {299: "Very Poor", 400: "Poor", 600: "Fair", 660: "Good", 780: "Excellent"}
for key, value in mapper.items():
    loan_df.loc[loan_df.fico > key, "credit_rating"] = value

loan_df["credit_rating"] = loan_df.credit_rating.astype("category")
# loan_df.info()

# Check the distribution of 'credit_rating' scores
loan_df.credit_rating.value_counts()
loan_df.credit_rating.value_counts(normalize=True)

# Create an 'int_rate_type' column
loan_df.loc[loan_df.int_rate > 0.12, "int_rate_type"] = "High"
loan_df.loc[loan_df.int_rate <= 0.12, "int_rate_type"] = "Low"
loan_df["int_rate_type"] = loan_df.int_rate_type.astype("category")
loan_df.info()

# Visualize the counts for 'purpose', 'credit_rating', and 'int_rate_type'
p_plot = loan_df.purpose.value_counts().to_frame()
p_plot.plot.pie(subplots=True, title="Number of Loans by Loan Purpose", legend=None)
plt.show()

cr_plot = loan_df.credit_rating.value_counts().to_frame()
cr_plot.plot.pie(subplots=True, title="Number of Loans by FICO Credit Rating", legend=None)
plt.show()

ir_plot = loan_df.int_rate_type.value_counts().to_frame()
ir_plot.plot.pie(subplots=True, title="Number of Loans by Interest Rate Type", legend=None)
plt.show()

# Is there a relationship between annual income and debt to income?
loan_df.plot.scatter(y="annual_inc", x="dti",
                     title="Relationship Between Annual Income and Debt to Income Ratio")
plt.show()

# Is there a relationship between annual income and interest rate?
loan_df.plot.scatter(x="annual_inc", y="int_rate",
                     title="Relationship Between Annual Income and Interest Rate")
plt.show()

# Save cleaned dataset for visualizing in Tableau
loan_df.to_csv("Data/bluebank_cleaned.csv", index=True)
