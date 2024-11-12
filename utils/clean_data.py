

def clean_data(data):
    """
    Clean the data by dropping missing values and unnecessary columns.
    """
    # Drop rows with missing values
    data = data.dropna()

    # Drop unnecessary columns (adjust based on your dataset)
    data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], errors='ignore')

    return data