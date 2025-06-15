# 🧠 Breast Cancer Detection using Deep Transfer Learning

This project presents a robust **dual-branch deep learning approach** that leverages both **histopathological** and **thermal breast images** for accurate breast cancer classification. By combining texture-rich histopathological data with thermal imaging insights, the model aims to improve diagnostic precision in identifying **malignant** and **benign** cases.

---

## 🔍 Overview

- 🔬 **Histopathological Images**: High-resolution tissue slides (BreakHis Dataset)
- 🌡️ **Thermal Images**: Simulated thermal views derived from pathology
- 🧠 **Model**: Dual-Branch CNN trained independently on both modalities and combined for final prediction
- 📊 **Evaluation**: Accuracy, confusion matrix, training curves

---

## 🗂️ Project Structure

├── src/ # Core scripts
│ ├── train.py # Training logic
│ ├── test.py # Testing logic
│ ├── model.py # Dual-branch CNN architecture
│ ├── dataset.py # Custom dataset loader
│ └── common_transforms.py # Image preprocessing
│
├── data/
│ ├── raw/ # Original histopathology images
│ └── thermal/ # Converted thermal images
│
├── outputs/
│ └── output.png # Sample output image
│
├── confusion matrix/
│ ├── confusion_matrix.png
│ └── confusion_matrix_old.png
│
├── training curves/
│ ├── training_curves.png
│ └── training_curves_old.png
│
├── dual_branch_cnn.pth # Trained model weights
├── dual_branch_cnn1.pth # Backup model weights
├── requirments.txt
└── README.md


---

## 🧪 Technologies Used

- Python
- PyTorch
- NumPy, OpenCV
- Matplotlib, Seaborn
- Scikit-learn

---

## 📌 Visualizations

### 📈 Training Curve

![Training Curve](training%20curves/training_curves.png)

### 📉 Confusion Matrix

![Confusion Matrix](confusion%20matrix/confusion_matrix.png)

---

## 🚀 How to Run

If u dont want to train it again and simply use the pretrained model my me then simply run the [python -m src.test] while the virtual environment is activate with all packages are installed which are required

1. **Install dependencies**
   ```bash [ before running the code scripts the virtual environment must to activated ]
   pip install -r requirments.txt
2. Train the model
python -m src.train

3. Test the model
python -m src.test

4. Evaluate and Visualize
python -m src.plot_training [ for ploting the training curve ]
python -m src.evaluate [ for evaluating the trained model ]
View confusion matrix: confusion_matrix/confusion_matrix.png
View training curve: training_curves/training_curves.png

---

📁 Dataset:

BreakHis Dataset (Histopathological images): Kaggle / BreakHis

Thermal images are derived during preprocessing using custom thermal_conversion.py.

Note: Dataset not included in repo due to size (4GB). Download and place under data/raw/.

---

⚙️ Features:

Dual-pathway processing of different image modalities

Simulated thermal generation pipeline

Modular code for easy training and testing

Visualization tools for debugging and evaluation

---

📈 Results:

Achieved high classification accuracy on both benign and malignant classes.

Dual-branch model outperformed single-branch baselines.

---

🤝 Contributing:

Pull requests are welcome. If you spot bugs or have suggestions for improvements, feel free to open an issue.

---

📄 License:

This project is licensed under the MIT License.

---

🙋‍♂️ Author

Veekshith H R

💼 LinkedIn:www.linkedin.com/in/veekshith-h-r-25a110248

💻 GitHub:https://github.com/hrveekshith

📧 Email: veekshithhrveekshithhr@gamil.com
