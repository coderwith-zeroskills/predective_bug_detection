import pandas as pd
df = pd.read_csv("jira_stories.csv")
print(df.shape)         # should print (500, 6)
print(df.columns.tolist())  # should list all 6 columns
print(df.head(3))