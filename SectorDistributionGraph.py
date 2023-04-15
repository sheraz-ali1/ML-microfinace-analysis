import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("kiva_loans.csv")

# group by
sector_loan = df.groupby('sector')['loan_amount'].mean().reset_index()

# sort
sector_loan = sector_loan.sort_values('loan_amount', ascending=False)

# create a bar chart to visualize the average loan amount per sector
plt.figure(figsize=(14, 8))
sns.barplot(x='loan_amount', y='sector', data=sector_loan, palette='Blues_r')
plt.title("Average Loan Amount per Sector", fontsize=14, fontweight='bold')
plt.xlabel("Average Loan Amount (USD)")
plt.ylabel("Sector")
plt.show()
