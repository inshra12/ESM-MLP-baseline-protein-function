import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
#Step 1:Load Data
def load_data(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)
    # Fill missing values with empty strings
    df.fillna('', inplace=True)
    # Combine MF, CC, BP GO terms into one list per protein
    df['Labels'] = df[['MF', 'CC', 'BP']].agg(';'.join, axis=1)
    df['Labels'] = df['Labels'].apply(lambda x: list(set(x.split(';')) if x else []))
    # Show first few rows
    print("First protein entry:\n", df.iloc[0])
    return df
#Step 2: Split Data
def split_data(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

#Step 3 Encode Labels
def encode_labels(label_lists):
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(label_lists)
    return encoded, mlb