# 🧠 Breast Cancer Detection Using Deep Transfer Learning

A multimodal deep learning system that detects breast cancer by analyzing both **thermal** and **histopathological** images. This project combines domain knowledge and powerful transfer learning techniques to enhance diagnostic accuracy.

![Dual Branch Model Output](outputs/output.png)

---

## 📁 Folder Structure

📦 breast_cancer_detection/
├── src/ # Source code: models, training, testing
├── data/ # 'raw' & 'thermal' data folders
├── outputs/ # Model output visuals
├── confusion matrix/ # Confusion matrix plots
├── training curves/ # Loss & accuracy graphs
├── testing_data/ # Samples used during inference
├── dual_branch_cnn*.pth # Trained model weights
├── training_history.pkl # Training metadata
├── requirements.txt # Python dependencies
├── env/ # (Optional) Virtual environment folder
└── README.md # You're here!

yaml
Copy
Edit

---

## ✅ Features

- Dual-branch architecture for fusion of thermal and histopathological data
- Custom dataset loaders and preprocessing scripts
- Transfer learning with pretrained CNNs
- Visual training/evaluation tracking
- Evaluation metrics and confusion matrix
- Easily test on new input images

---

## 🛠️ Tech Stack

- Python
- PyTorch
- OpenCV, NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## 🧠 Model Architecture

- Each input type (thermal and histopathology) is passed through its own CNN (based on ResNet).
- Feature vectors from both branches are **concatenated** and passed through dense layers.
- Final output: Binary classification (Malignant / Benign).

---

## 📊 Results

| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| Thermal       | ~92%     | 90.2%     | 93.1%  | 91.6%    |
| Histological  | ~95%     | 94.1%     | 96.2%  | 95.1%    |
| **Dual-Branch** | **97.2%** | **96.5%** | **97.9%** | **97.2%** |

📌 See:
- `confusion matrix/confusion_matrix.png`
- `training curves/training_curves.png`

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
1.git clone https://github.com/hrveekshith/Breast_cancer_detection_using_deep_tranfer_learning.git
cd Breast_cancer_detection_using_deep_tranfer_learning
2. Install Dependencies
pip install -r requirements.txt
3. Run Inference
python src/test.py --img_path testing_data/SOB_B_TA-14-16184-200-002.png
4. Train the Model (Optional)
python src/train.py
🔔 You must download the BreakHis dataset manually and place it inside the data/raw folder.
Dataset link

📌 Visualizations
Training Curve:

Confusion Matrix:

🤝 Contributing
Pull requests are welcome. If you spot bugs or have suggestions for improvements, feel free to open an issue.

📄 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Veekshith Gowda H R
🔗 GitHub Profile

