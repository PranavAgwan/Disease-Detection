# 🧬 Disease Detection using CNN and Random Forest Algorithm

## 🧠 Introduction
This project focuses on creating a **disease detection system** capable of identifying four diseases:  

- **Brain Tumor**  
- **Breast Cancer**  
- **Covid-19**  
- **Pneumonia**  

The project combines **deep learning (CNN)** and **machine learning (Random Forest)** techniques to predict the diseases from medical datasets. The workflow is divided into two main phases:  

1. **Training Phase** – Training all models using appropriate algorithms.  
2. **Deployment Phase** – Deploying the trained models in a **Flask web application** to provide disease prediction and solutions to users.  

The system is designed to be accessible, allowing users to get predictions and solutions regardless of location, providing a practical tool especially in pandemic scenarios or areas with limited medical access.

> 📝 All analysis, model training, and intermediate results are available in the Jupyter Notebook: **`Disease_Detection.ipynb`**.

---

## ⚙️ Methodology

### 1️⃣ Key Technologies
- **Convolutional Neural Network (CNN)**  
  - Used for Brain Tumor, Covid, and Pneumonia detection.  
  - CNNs are specialized deep learning networks capable of recognizing patterns and features in structured data such as images.  
- **Random Forest Classifier**  
  - Used for Breast Cancer prediction.  
  - Ensemble-based supervised ML algorithm that combines multiple decision trees for improved predictive accuracy.  
- **Flask, HTML, CSS**  
  - Flask was used to deploy the trained models as a web application.  
  - HTML and CSS were used to design the front-end user interface.

---

### 2️⃣ Scope and Importance
- Provides a **web-based interface** for disease detection, accessible globally.  
- Helps users check for diseases and receive suggested solutions **without visiting hospitals**, reducing healthcare bottlenecks.  
- Particularly relevant for scenarios like **COVID-19**, where access to medical professionals may be limited.

---

### 3️⃣ Implementation / Design

#### Phase 1 – Model Training
1. **CNN Models for Image-based Diseases**
   - **Brain Tumor, Covid, Pneumonia**
   - Input: Image datasets
   - Process: Images preprocessed and fed into CNN architectures.
   - Output: Trained CNN models capable of predicting the presence or absence of disease.  

2. **Random Forest for Breast Cancer**
   - Input: CSV dataset of 569 breast cancer entries
   - Process: Dataset split into training and testing sets, Random Forest model trained on features.
   - Output: Trained classifier predicting malignant vs. benign tumors.

#### Phase 2 – Model Deployment
- Trained models saved as files (one for each disease).  
- Flask framework used to create a **local web application**.  
- **Front-end** created using HTML and CSS to provide a user-friendly interface.  
- Users can:
  1. Navigate to a specific disease section from the main page.  
  2. Enter basic information (name, age, gender, contact).  
  3. Upload medical data (e.g., MRI images for Brain Tumor).  
  4. Receive **predictions and suggested solutions** in seconds.  

---

## 📊 Testing, Results & Analysis

| Disease         | Dataset Details | Accuracy |
|-----------------|----------------|---------|
| Brain Tumor     | Training: 386 images, Validation: 100, Testing: 20 | 86.01% |
| Breast Cancer   | 569 entries (CSV dataset) | 94.73% |
| Covid-19        | Training: 282 images, Validation: 80, Testing: 30 | 95.6% |
| Pneumonia       | Training: 5219 images, Validation: 19, Testing: 625 | 94.47% |

**Key Notes:**
- CNN models achieved high accuracy for image-based diseases.  
- Random Forest performed well for structured data (breast cancer).  
- All models are integrated into a **Flask web app** for real-time predictions.  
- Local host interface screenshots and further results can be added to the repository for visual reference.

---

## 🧾 Conclusion
This project successfully demonstrates an **end-to-end AI/ML disease detection pipeline**, from model training to deployment as a web application.  

**Highlights:**
- **Multiple algorithms** used for different disease types (CNN for image-based, Random Forest for structured data).  
- **Web deployment** allows non-expert users to access predictions remotely.  
- **High accuracy** across all diseases (86–95%).  
- Provides a **solution recommendation system** alongside disease detection.  

This framework can be extended to additional diseases, larger datasets, or a full-fledged online deployment for broader healthcare accessibility.

---

## 🛠️ Tech Stack

| Tool / Technology | Purpose |
|------------------|---------|
| **Python** | Programming language for model training and backend |
| **Jupyter Notebook** | Integrated environment for code, analysis, and reporting |
| **CNN (Keras / TensorFlow)** | Deep learning for image-based disease detection |
| **Random Forest (Scikit-learn)** | Supervised machine learning for structured data |
| **Flask** | Python micro-framework for web deployment |
| **HTML & CSS** | Front-end interface for web application |
| **Pandas & NumPy** | Data preprocessing and manipulation |
| **Matplotlib & Seaborn** | Visualizations for EDA and model evaluation |

---
