# 💫 PriceGenie AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge\&logo=python\&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-orange?style=for-the-badge)](https://github.com/SQUADRON-LEADER/PriceGenie-AI)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**🤖 The Smart Way to Master Product Pricing!**

PriceGenie AI helps businesses make **data-driven pricing decisions** using **machine learning**. With a sleek **Streamlit dashboard**, a powerful **Flask backend**, and top-tier **ML models (LightGBM, XGBoost, CatBoost)** — it gives real-time pricing insights to boost profits and stay competitive. 💹

---

## 🎬 Demo

<img width="1919" height="868" alt="Screenshot 2025-10-09 162818" src="https://github.com/user-attachments/assets/aed54e8e-d391-4df4-ab2d-702a761c002c" />

<img width="1912" height="854" alt="Screenshot 2025-10-09 162839" src="https://github.com/user-attachments/assets/7c6cfe3c-6159-458f-9ef3-855d21b8e028" />

<img width="1919" height="871" alt="Screenshot 2025-10-09 162851" src="https://github.com/user-attachments/assets/5497d8c7-1ea9-4850-9fbb-fb4b3bcb02b1" />


<img width="1919" height="862" alt="Screenshot 2025-10-09 162858" src="https://github.com/user-attachments/assets/e79f36e3-58f2-42e1-9e84-fa498eafb888" />

<img width="1919" height="860" alt="Screenshot 2025-10-09 162906" src="https://github.com/user-attachments/assets/9f59e6f8-451d-4f62-a07b-072aebb4490c" />

<img width="1919" height="860" alt="Screenshot 2025-10-09 162913" src="https://github.com/user-attachments/assets/849cc862-6471-42c0-a7ac-b82709a2094c" />

---

## ✨ Key Features

🚀 **Multi-Model Powerhouse** — Uses LightGBM, XGBoost, and CatBoost for accurate predictions.
📊 **Interactive Dashboard** — Explore trends, visualize metrics, and get instant insights.
🧠 **AI-Powered API** — Flask REST API for seamless integration with your systems.
📈 **Performance Metrics** — Compare models and track historical improvements.
⚡ **Real-time Predictions** — Get instant pricing suggestions for your products.
🛠️ **Simple Setup** — Easy to install, run, and extend.

---

## 🗂️ Project Structure

```
PriceGenie-AI/
├── models/                # Trained model files
├── python-backend/        # Flask API & backend logic
├── streamlit_app.py       # Streamlit dashboard UI
├── Amazon_ML_Multi_Algorithm_Training.ipynb  # Model training notebook
├── requirements.txt
├── start-backend.bat
├── start-streamlit.bat
└── README.md
```

📁 *Folders may evolve as development continues.*

---

## 🧩 Tech Stack

| Layer            | Technology                                   |
| ---------------- | -------------------------------------------- |
| 🎨 Frontend      | Streamlit                                    |
| ⚙️ Backend       | Flask (Python)                               |
| 🤖 ML Models     | LightGBM · XGBoost · CatBoost · scikit-learn |
| 📊 Visualization | Plotly                                       |
| 📚 Data Handling | pandas · numpy                               |

---

## 🪄 Quick Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SQUADRON-LEADER/PriceGenie-AI.git
cd PriceGenie-AI
```

### 2️⃣ Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r python-backend/requirements.txt
```

### 4️⃣ Run the App

**Windows:**

```bash
start-streamlit.bat   # Starts Streamlit Dashboard
start-backend.bat     # Starts Flask API
```

**macOS/Linux:**

```bash
streamlit run streamlit_app.py --server.port 8502
cd python-backend && python app.py
```

### 🌐 Access Points

* 📊 Dashboard → `http://localhost:8502`
* 🧠 API → `http://localhost:5000`
* ❤️ Health Check → `http://localhost:5000/health`

---

## 🧠 How It Works

### 🎛️ Streamlit Dashboard

* Upload product data 📂
* Explore model results 🔍
* Get recommended prices 💰
* Export insights for business use 📤

### 🔌 Flask API Example

```json
POST /predict
{
  "product_id": "SKU123",
  "features": { "cost": 10.5, "category": "electronics", "demand_score": 0.8 }
}
```

**Response:**

```json
{
  "predicted_price": 14.99,
  "model": "ensemble",
  "confidence": 0.87
}
```

---

## 📘 Model Training

* All training experiments are in: `Amazon_ML_Multi_Algorithm_Training.ipynb`
* Models are saved in `/models/` for serving 🧾

💡 Steps to Train:

1. Clean your data 🧹
2. Feature engineering 🔧
3. Train models 🧠
4. Evaluate performance 📈
5. Export models 📦

---

## ⚙️ Configuration

* Store secrets & keys in a `.env` file 🔒
* Use environment variables for deployment 🌍

---

## 🧪 Testing

Recommended with `pytest` ✅

* API route tests 🧩
* Model output validation 🧠
* Data preprocessing checks 🧾

---

## ☁️ Deployment

Options to deploy easily:

* 🐳 Docker containers
* ☁️ Streamlit Cloud / Render / AWS
* 🔄 CI/CD with GitHub Actions

---

## 🤝 Contributing

💡 Want to help improve PriceGenie?

1. Fork the repo 🍴
2. Create a feature branch 🌱
3. Submit a pull request ✨

📝 Please include clear commit messages and update docs as needed.

---

## 📜 License

MIT License

Copyright (c) 2025 Aayush Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 🌟 Acknowledgements

🙏 Thanks to these amazing tools:

* LightGBM · XGBoost · CatBoost · scikit-learn
* Streamlit · Flask · Plotly

---

💬 *Let PriceGenie AI make your pricing smarter, faster, and fairer!* ⚡









