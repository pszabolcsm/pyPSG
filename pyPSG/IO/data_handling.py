from scipy.io import savemat

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
