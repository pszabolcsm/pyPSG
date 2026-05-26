from scipy.io import savemat
import pandas as pd

def save_data(data, output_path):
    """
         Save a dictionary of data to a MATLAB .mat file.

         :param data: A dictionary containing variable names as keys and the corresponding data as values.
         :type data: dict
         :param output_path: Path where the .mat file will be saved. If it doesn't end with '.mat', it will be added automatically.
         :type output_path: str

         :return: None
    """
    # Ensure the output file has the correct .mat extension
    if not output_path.endswith('.mat'):
        output_path += '.mat'
    
    # Save the data to the .mat file
    savemat(output_path, data)


def save_data_mat(data, output_path):
    """
         Save a dictionary of data to a MATLAB .mat file.

         :param data: A dictionary containing variable names as keys and the corresponding data as values.
         :type data: dict
         :param output_path: Path where the .mat file will be saved. If it doesn't end with '.mat', it will be added automatically.
         :type output_path: str

         :return: None
    """
    # Ensure the output file has the correct .mat extension
    if not output_path.endswith('.mat'):
        output_path += '.mat'
    
    # Save the data to the .mat file
    savemat(output_path, data)
    
def save_data_csv(data, output_path):
    """
        Save dictionary contents into CSV files.

        Parameters
        ----------
        data : dict
            Dictionary containing biomarker data.
        output_dir : str
            Directory where CSV files will be saved.
    """
    
    if not output_path.endswith(".csv"):
        output_path += ".csv"

    if isinstance(data, pd.DataFrame):

        data.to_csv(output_path, index=False)

    elif isinstance(data, dict):

        df = pd.DataFrame.from_dict(data)
        df.to_csv(output_path, index=False)

    else:

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
