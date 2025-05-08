# Give me a function to visualise time series data given a pd dataframe

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pywatts.modules.generation.anomaly_generation_module import AnomalyGeneration




def preprocess_data(file_path, selected_building_id=None):
    """
    Preprocesses the dataset by sorting, handling missing values, and extracting data for a specific building.
    
    Args:
        file_path: Path to the CSV file containing the dataset
        selected_building_id: Optional - specific building ID to extract (uses first building if None)
        
    Returns:
        Preprocessed DataFrame with meter readings ready for anomaly injection
    """
    # Load the dataset
    dataset = pd.read_csv(file_path)
    
    # Sort by building_id and timestamp in ascending order
    dataset = dataset.sort_values(by=['building_id', 'timestamp'], ascending=[True, True])
    
    # Perform median imputation to handle missing values
    def median_imputation(df):
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Find all unique building IDs
        unique_building_ids = df_copy['building_id'].unique()
        
        for i in unique_building_ids:
            # For each building ID, find the median of the 'meter_reading' column
            median_value = df_copy[df_copy['building_id'] == i]['meter_reading'].median()
            
            # Fill missing values in the 'meter_reading' column with the median value
            df_copy.loc[(df_copy['building_id'] == i) & (df_copy['meter_reading'].isnull()), 'meter_reading'] = median_value
        
        return df_copy
    
    # Apply median imputation to the dataset
    dataset = median_imputation(dataset)
    
    # Extract the unique building IDs
    unique_building_ids = dataset['building_id'].unique()
    
    # Select building ID - either the specified one or the first one in the list
    building_id = selected_building_id if selected_building_id is not None else unique_building_ids[0]
    
    # Extract data for the selected building
    test_dataset = dataset[dataset['building_id'] == building_id].copy()
    
    # Convert the 'timestamp' column to datetime format
    test_dataset['timestamp'] = pd.to_datetime(test_dataset['timestamp'])
    
    # Drop unnecessary columns and reset index
    test_dataset = test_dataset.drop(columns=['building_id'])
    test_dataset = test_dataset.reset_index(drop=True)
    
    return test_dataset

class PowerAnomalyGeneration(AnomalyGeneration):
    """
    Module to define specific anomalies to be inserted into a power time series.
    """

    def _anomaly_type1(self, target, indices, lengths, k=0):
        """
        Anomaly type 1 that drops the power time series values to a negative value potentially followed by zero values
        before adding the missed sum of power to the end of the anomaly.
        """
        for idx, length in zip(indices, lengths):
            if length <= 2:
                raise Exception("Type 1 power anomalies must be longer than 2.")
            else:
                # WARNING: This could lead to a overflow quite fast?
                energy_at_start = target[:idx].sum() + k
                energy_at_end = target[:idx + length].sum() + k
                target[idx] = -1 * energy_at_start  # replace first by negative peak
                target[idx + 1:idx + length - 1] = 0  # set other values to zero
                target[idx + length - 1] = energy_at_end  # replace last with sum of missing values + k
        return target

    def _anomaly_type2(self, target, indices, lengths, softstart=True):
        """
        Anomaly type 2 that drops the power time series values to potentially zero and adds the missed sum of power to
        the end of the anomaly.
        """
        for idx, length in zip(indices, lengths):
            if length <= 1:
                raise Exception("Type 2 power anomalies must be longer than 1.")
            else:
                if softstart:
                    r = np.random.rand()
                    energy_consumed = target[idx:idx + length].sum()
                    target[idx] = r * target[idx]
                    target[idx + 1:idx + length - 1] = 0
                    target[idx + length - 1] = energy_consumed - target[idx]
                else:
                    energy_consumed = target[idx:idx + length].sum()
                    target[idx:idx + length - 1] = 0
                    target[idx + length - 1] = energy_consumed
        return target

    def _anomaly_type3(self, target, indices, lengths,
                       is_extreme=False, range_r=(0.01, 3.99), k=0):
        """
        Anomaly type 3 that creates a negatives peak in the power time series.
        """
        for idx, length in zip(indices, lengths):
            if length > 1:
                raise Exception("Type 3 power anomalies can't be longer than 1.")
            else:
                if is_extreme:
                    energy_consumed = target[:idx].sum()
                    target[idx] = -1 * energy_consumed - k
                else:
                    r = np.random.uniform(*range_r)
                    target[idx] = -1 * r * target[idx - 1]
        return target

    def _anomaly_type4(self, target, indices, lengths, range_r=(0.01, 3.99)):
        """
        Anomaly type 4 that creates a positive peak in the power time series.
        """
        for idx, length in zip(indices, lengths):
            if length > 1:
                raise Exception("Type 4 power anomalies can't be longer than 1.")
            else:
                r = np.random.uniform(*range_r)
                target[idx] = r * target[idx - 1]
        return target


def inject_anomalies_into_data(df, type, num_anomalies, anomaly_length):

    df_with_anomalies = df.copy()

    target = df_with_anomalies['meter_reading'].values.copy()

    anomaly_gen = PowerAnomalyGeneration()

    if type == 1 and anomaly_length <= 2:
        raise ValueError("Type 1 anomalies must have length > 2")
    elif type == 2 and anomaly_length <= 1:
        raise ValueError("Type 2 anomalies must have length > 1")
    elif (type == 3 or type == 4) and anomaly_length != 1:
        raise ValueError(f"Type {type} anomalies must have length = 1")
    
    safe_upper_limit = len(target) - anomaly_length - 5
    safe_lower_limit = 5

    if safe_upper_limit <= safe_lower_limit:
        raise ValueError("Data series too short for specified anomaly length")
    
    indices = np.random.choice(range(safe_lower_limit, safe_upper_limit), 
                               num_anomalies, replace=False)
    indices = sorted(indices)  # Sort to ensure sequential processing
    
    # Create lists of same length for indices and lengths
    lengths = [anomaly_length] * num_anomalies
    
    # Create anomaly flags (1 where anomalies will be inserted)
    anomaly_flags = np.zeros_like(target, dtype=int)
    
    # Apply the specified anomaly type
    if type == 1:
        # Type 1: Negative peak, zeros, compensating positive value
        target = anomaly_gen._anomaly_type1(target, indices, lengths)
        
    elif type == 2:
        # Type 2: Zeros with energy conservation at end
        target = anomaly_gen._anomaly_type2(target, indices, lengths, softstart=True)
        
    elif type == 3:
        # Type 3: Negative spike
        target = anomaly_gen._anomaly_type3(target, indices, lengths, 
                                           is_extreme=False, range_r=(1.5, 4.0))
        
    elif type == 4:
        # Type 4: Positive spike
        target = anomaly_gen._anomaly_type4(target, indices, lengths, range_r=(1.5, 4.0))
        
    else:
        raise ValueError("Anomaly type must be between 1 and 4")
    
    # Mark the anomalous regions in the anomaly flags
    for idx, length in zip(indices, lengths):
        anomaly_flags[idx:idx+length] = 1
    
    # Update the DataFrame
    df_with_anomalies['meter_reading'] = target  # Replace with anomalous readings
    df_with_anomalies['is_anomaly'] = anomaly_flags
    
    return df_with_anomalies


def plot_anomaly_time_series(df_original, df_with_anomalies, title="Power Consumption with Anomalies", 
                             xlabel="Time", ylabel="Meter Reading"):
    """
    Plots time series data comparing original readings with anomalous readings.

    Parameters:
    df_original (pd.DataFrame): DataFrame containing original time series data with 'timestamp' and 'meter_reading' columns.
    df_with_anomalies (pd.DataFrame): DataFrame containing time series with anomalies, must have 'timestamp', 
                                      'meter_reading', and 'is_anomaly' columns.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(15, 6))
    
    # Plot original data using timestamps
    plt.plot(df_original['timestamp'], df_original['meter_reading'], 
             color='blue', alpha=0.7, label='Original')
    
    # Plot data with anomalies using timestamps
    plt.plot(df_with_anomalies['timestamp'], df_with_anomalies['meter_reading'], 
             color='red', alpha=0.7, label='With Anomalies')
    
    # Highlight anomaly regions
    anomaly_points = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1]
    plt.scatter(anomaly_points['timestamp'], 
                anomaly_points['meter_reading'],
                color='red', s=50, marker='x', label='Anomalies')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format the x-axis to show dates appropriately
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()
    
    # Create an image file name based on the title
    image_file_name = title.replace(" ", "_").lower() + ".png"
    # Save the plot as an image file
    plt.savefig(image_file_name)
    plt.close()

if __name__ == "__main__":
    # Example usage
    file_path = 'train.csv'
    df_original = preprocess_data(file_path, selected_building_id=1)
    
    # Make a copy before injecting anomalies
    df = df_original.copy()
    print(df.head())
    
    # Inject anomalies
    df_with_anomalies = inject_anomalies_into_data(df, type=2, num_anomalies=5, anomaly_length=3)
    
    # Visualize the results using timestamps
    plot_anomaly_time_series(df_original, df_with_anomalies)



