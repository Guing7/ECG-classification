
# ECG Classification Project

This project provides a model for ECG classification using deep learning. The dataset used is the **Chapman University and Shaoxing People’s Hospital ECG Dataset**, available at [this link](https://figshare.com/collections/ChapmanECG/4560497/2).

## Dataset

The dataset used in this project is from the **Chapman University and Shaoxing People’s Hospital**. You can download the dataset by visiting the link above and then extract the `.7z` file to obtain the data in `.npy` format, which is used to train the ECG classification model.

The dataset is provided as a compressed file `ECGDataNPY.7z`. After downloading it, extract the contents to get the `.npy` dataset files.

### Steps to Extract the Dataset:
1. Download the `ECGDataNPY.7z` file.
2. Extract the file using your preferred archive manager (e.g., WinRAR, 7-Zip, etc.).
3. The extracted folder will contain the dataset in `.npy` format.

## Requirements

To run the project, make sure you have the following Python packages installed:

- Python 3.12
- `torch`
- `numpy`
- `pandas`
- `seaborn`
- `scikit-learn`
- `matplotlib`
- `openpyxl`
- `tqdm`
- `git-lfs`

You can install all the required packages by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. **Training the Model:**
   After setting up the environment and extracting the dataset, you can train the ECG classification model by running the following command:

```bash
python train_model.py
```

This will train the model using the provided dataset and save the best model as `best_fusion_model.pth` based on validation accuracy.

2. **Using the Pre-trained Model:**
   If you want to test the model's performance without training it yourself, you can use the pre-trained model (`best_fusion_model.pth`). To do this, run the following script to load the model and evaluate it on the test set:

```bash
python test_best_model.py
```

This script will load the `best_fusion_model.pth` file, which contains the best weights obtained from training, and perform evaluation on the test set, producing the test accuracy and confusion matrix.

## File Structure

The project directory should look something like this:

```
ECG-Classification-Project/
│
├── ECGDataNPY/                # Folder containing .npy dataset files
├── best_fusion_model.pth      # Pre-trained model
├── requirements.txt           # List of required Python packages
├── train_model.py             # Training script for the model
├── test_best_model.py         # Script to test the pre-trained model
└── README.md                  # Project documentation
```

## Notes

- The dataset is large, and we recommend using a machine with enough memory (RAM and GPU) to handle the data and model training.
- Make sure to have Git Large File Storage (LFS) installed if you're using Git to manage the large dataset files.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
