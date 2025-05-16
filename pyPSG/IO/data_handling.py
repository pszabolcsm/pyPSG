from scipy.io import savemat

def save_data(data, output_path):
    if not output_path.endswith('.mat'):
        output_path += '.mat'
        
    savemat(output_path, data)
