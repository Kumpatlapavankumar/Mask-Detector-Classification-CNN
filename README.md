# 😷 Mask Detector

A simple and interactive **Face Mask Detection** web app built with **Streamlit** and **TensorFlow**.  
It allows users to upload an image and detects whether the person is **wearing a mask** or **not** using a trained deep learning model.

---

## 📌 Features
- Upload images in JPG, JPEG, PNG, or WEBP format.
- Detects **With Mask** or **Without Mask**.
- Uses a pre-trained TensorFlow model stored on Google Drive.
- Clean and responsive Streamlit UI.

---

## 🛠 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Kumpatlapavankumar/mask-detector.git
cd mask-detector
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ (Optional) Install system dependencies
If running locally, make sure you have **OpenCV** and **Pillow** installed via `requirements.txt`.

---

## 🚀 Usage

Run the app locally:
```bash
streamlit run main.py
```

Once running, the app will open in your browser. Upload an image, and the model will predict whether the person is wearing a mask.

---

## 📂 Project Structure
```
.
├── main.py             # Main Streamlit application
├── requirements.txt    # Python dependencies
├── packages.txt        # Additional dependencies (optional for deployment)
└── README.md           # Documentation
```

---

## 📦 Model
- The model (`mask_detector_model.keras`) is stored on Google Drive and is automatically downloaded on first run.
- Update the `file_id` variable in the script with your model's Google Drive ID if you replace the model.

---

## 📝 License
This project is licensed under the MIT License.

---

## 👨‍💻 Author
**Pavankumar**  
💻 Powered by **TensorFlow** & **Streamlit**
