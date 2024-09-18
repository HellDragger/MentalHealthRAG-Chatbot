import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('../data/raw_data/CSV Files/context_response_train.csv')

# Display the first few rows
print(df.head())
