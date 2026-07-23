# 👤 Real-Time Face Recognition & Employee Attendance System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org)
[![RetinaFace](https://img.shields.io/badge/Detector-RetinaFace-success.svg)]()
[![FaceNet](https://img.shields.io/badge/Recognition-InceptionResNet-orange.svg)]()
[![Triplet Loss](https://img.shields.io/badge/Loss-Triplet-purple.svg)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)]()

> **An end-to-end AI-powered employee attendance system that detects, recognizes, and logs employees automatically using CCTV cameras, RetinaFace, InceptionResNet embeddings, cosine similarity, and a real-time Streamlit dashboard.**

---

# 📌 Project Overview

This project automates employee attendance using **computer vision** and **deep metric learning**, eliminating the need for manual attendance systems.

The system detects faces from a live CCTV feed, generates facial embeddings using a fine-tuned **InceptionResNet (FaceNet)** model, compares them against a stored employee database, and records attendance only after multiple successful recognitions to minimize false positives.

A real-time **Streamlit dashboard** provides live monitoring, attendance logs, and an intuitive interface for HR and security teams.

---

# ✨ Key Features

- Real-time Face Detection using RetinaFace
- Face Recognition using InceptionResNet (FaceNet)
- 512-D Face Embeddings
- Triplet Loss-based Metric Learning
- Cosine Similarity Matching
- 10-Frame Confirmation Logic
- Automatic Attendance Logging
- JSON Embedding Database
- Live Streamlit Dashboard
- Multi-person Recognition
- Unknown Face Detection
- Production-style Pipeline

---

# 🎯 Problem Statement

Traditional attendance systems suffer from several limitations:

- Manual attendance marking
- Proxy attendance
- Time-consuming verification
- Human errors
- No real-time monitoring

This project aims to provide an automated attendance solution that:

- Detects faces from CCTV footage
- Recognizes registered employees
- Prevents false attendance entries
- Handles multiple people simultaneously
- Displays attendance in real time

---

# 🏗️ System Architecture

```text
Live CCTV Feed
        │
        ▼
 RetinaFace Detection
        │
        ▼
 Face Cropping
        │
        ▼
 Resize (160×160)
        │
        ▼
 InceptionResNet
        │
        ▼
512-D Face Embedding
        │
        ▼
 Cosine Similarity
        │
        ▼
10-Frame Confirmation
        │
        ▼
 Attendance Logging
        │
        ▼
 Streamlit Dashboard
```

---

# 🛠️ Tech Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Face Detection | RetinaFace |
| Face Recognition | InceptionResNet (FaceNet) |
| Learning Method | Triplet Loss |
| Similarity Metric | Cosine Similarity |
| Database | JSON |
| Dashboard | Streamlit |
| Visualization | t-SNE |

---

# 📊 Project Results

| Metric | Value |
|---------|------:|
| Face Identification Accuracy | **95%** |
| Face Detector | RetinaFace |
| Recognition Model | InceptionResNet |
| Embedding Size | 512 |
| Similarity Metric | Cosine Similarity |
| Attendance Validation | 10 Consecutive Matches |
| Dashboard | Streamlit |

---

# 📂 Dataset Preparation

Before the system can recognize employees, a reference face database must be created.

Each employee is registered with **multiple face images** captured under different conditions such as varying lighting, facial expressions, and viewing angles.

These images are processed to generate a robust facial representation for each employee.

### Dataset Summary

| Property | Description |
|----------|-------------|
| Input | Employee Face Images |
| Multiple Images per Person | ✅ Yes |
| Image Size (Recognition) | 160 × 160 |
| Output | Face Embeddings |
| Database Format | JSON |

---

# 🧠 Step 1 — Face Embedding Generation

Instead of storing raw face images, the system stores **512-dimensional face embeddings**.

Each registered employee image is passed through a fine-tuned **InceptionResNet (FaceNet)** model to generate a unique numerical representation.

```text
Employee Images
        │
        ▼
 Face Preprocessing
        │
        ▼
 InceptionResNet
        │
        ▼
512-D Embedding
        │
        ▼
 JSON Database
```

To improve recognition accuracy, embeddings from multiple images of the same employee are averaged to create a stable reference representation.

---

# 🎯 Why Face Embeddings?

A face image contains thousands of pixel values, making direct comparison inefficient.

Instead, the model converts each face into a compact **512-dimensional embedding**, where:

- Similar faces produce embeddings that are close together.
- Different faces produce embeddings that are far apart.

This makes recognition faster, scalable, and suitable for adding new employees without retraining the model.

---

# 🏋️ Step 2 — Model Training

The face recognition model is based on **InceptionResNet**, pretrained on the **VGGFace2** dataset and fine-tuned for the employee recognition task.

### Training Configuration

| Parameter | Value |
|-----------|------|
| Backbone | InceptionResNet |
| Pretrained Dataset | VGGFace2 |
| Embedding Size | 512 |
| Optimizer | Adam |
| Loss | Triplet Loss + Cross Entropy |
| Data Augmentation | Rotation, Zoom, Flip |

---

# 📌 Why Triplet Loss?

Traditional classification models require retraining whenever a new employee is added.

Triplet Loss solves this limitation by learning a feature space instead of fixed class labels.

```text
Anchor Image

Positive Image (Same Person)

Negative Image (Different Person)

↓

Triplet Loss

↓

Learn Better Face Embeddings
```

This allows new employees to be added simply by generating and storing their embeddings—no model retraining required.

---

# 📹 Step 3 — Face Detection (RetinaFace)

During inference, the system processes live CCTV video.

Each frame is passed through **RetinaFace**, which detects all visible faces.

```text
Live CCTV Frame
        │
        ▼
 RetinaFace
        │
        ▼
 Face Bounding Boxes
```

### Why RetinaFace?

- High detection accuracy
- Handles side faces
- Detects small faces
- Robust under different lighting conditions
- Performs better than general object detectors for face-specific tasks

Faces smaller than **80 × 80 pixels** are ignored to avoid unreliable recognition.

---

# 🔄 Step 4 — Face Recognition Pipeline

Each detected face follows the recognition pipeline:

```text
Detected Face
        │
        ▼
 Crop Face
        │
        ▼
 Resize (160×160)
        │
        ▼
 InceptionResNet
        │
        ▼
512-D Embedding
        │
        ▼
 Cosine Similarity
        │
        ▼
 Best Matching Employee
```

The generated embedding is compared against all stored employee embeddings using **Cosine Similarity**.

The employee with the highest similarity score above the threshold is selected as the predicted identity.

---

# 📏 Cosine Similarity Matching

Instead of comparing raw images, the system compares face embeddings.

```text
Employee Database

↓

Embedding A

Embedding B

Embedding C

↓

Cosine Similarity

↓

Highest Similarity Score

↓

Recognized Employee
```

This approach is computationally efficient and robust to variations in lighting, pose, and facial expressions.

---

# 📈 Embedding Validation

To evaluate the quality of learned embeddings, **t-SNE** is used for visualization.

The 512-dimensional embeddings are projected into a 2D space.

A well-trained model produces:

- Tight clusters for the same employee
- Clear separation between different employees

This provides a visual confirmation that the model has learned meaningful facial representations.

---
# ✅ Step 5 — Attendance Confirmation Logic

A single face match is not sufficient for reliable attendance, as CCTV footage may contain motion blur, occlusions, or poor lighting.

To improve reliability, the system confirms an employee's identity only after **10 consecutive successful matches** across processed frames.

```text
Face Detected
      │
      ▼
Generate Embedding
      │
      ▼
Cosine Similarity Match
      │
      ▼
Match Count = 1
      │
      ▼
...
      │
      ▼
Match Count = 10
      │
      ▼
Attendance Confirmed
```

### Why 10 Consecutive Matches?

- Reduces false positives
- Avoids accidental recognition
- Improves reliability in real CCTV environments
- Prevents attendance from being marked due to a single noisy frame

---

# 🚫 Unknown Face Handling

If a detected face does not meet the similarity threshold with any registered employee, it is classified as **Unknown**.

The system:

- Does not mark attendance
- Ignores unauthorized identities
- Stores unknown face images separately for manual review (optional)

This improves security while preventing incorrect attendance entries.

---

# ⏱️ Performance Optimizations

To ensure smooth real-time performance, several optimizations were implemented:

| Optimization | Purpose |
|--------------|---------|
| Process every 10th frame | Reduce computation |
| Ignore faces smaller than 80×80 | Avoid poor-quality recognition |
| Pre-computed embeddings | Faster matching |
| JSON embedding database | Lightweight storage |
| Cooldown mechanism | Prevent duplicate attendance |

---

# 📊 Streamlit Dashboard

A Streamlit dashboard was developed to provide an easy-to-use interface for HR and security teams.

### Dashboard Features

- 🎥 Live CCTV video feed
- 👤 Face bounding boxes with employee names
- ✅ Real-time attendance status
- 📋 Attendance log with timestamps
- 📅 Daily attendance records

```text
+-------------------------------------------+
|           Live Camera Feed                |
|                                           |
|  [Employee Name]  [Bounding Box]          |
|                                           |
+-------------------------------------------+

Today's Attendance

✔ John Doe      09:02 AM
✔ Alice Smith   09:05 AM
✔ Mike Brown    09:08 AM
```

The dashboard allows non-technical users to monitor attendance without interacting with the underlying AI models.

---

# 📈 Results

| Metric | Value |
|---------|------:|
| Face Identification Accuracy | **95%** |
| Recognition Method | Face Embeddings |
| Embedding Size | 512 |
| Detection Model | RetinaFace |
| Recognition Model | InceptionResNet |
| Similarity Metric | Cosine Similarity |
| Dashboard | Streamlit |
| Attendance Validation | 10 Consecutive Matches |

---

# 💡 Key Achievements

- Built a complete end-to-end face recognition system
- Automated employee attendance using CCTV
- Implemented deep metric learning with Triplet Loss
- Generated robust 512-dimensional face embeddings
- Reduced false positives using 10-frame confirmation
- Developed a real-time monitoring dashboard
- Designed a scalable employee registration pipeline

---

# ⚡ Challenges & Solutions

| Challenge | Solution |
|------------|----------|
| Different lighting conditions | Data augmentation during training |
| Side faces and occlusions | RetinaFace detector |
| False recognitions | 10-frame confirmation logic |
| New employee onboarding | Embedding-based registration (no retraining) |
| Duplicate attendance entries | Cooldown mechanism |
| Real-time performance | Frame skipping and optimized inference |

---

# 🚀 Future Improvements

Some possible enhancements include:

- Face anti-spoofing detection
- Face mask recognition
- Multi-camera synchronization
- Database integration (MySQL/PostgreSQL)
- Cloud deployment
- REST API using FastAPI
- Email/SMS attendance notifications
- Mobile attendance dashboard

---

# 📄 License

This project is released under the **MIT License**.

---

# 👩‍💻 Author

**Namrata Patel**

AI/ML Engineer | Computer Vision | Deep Learning

📧 Email: Namratapatel091@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/namratapatel9027/

💼 Portfolio: https://codebasics.io/portfolio/Namrata-patel

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

**Built with Python, PyTorch, RetinaFace, InceptionResNet, and Streamlit ❤️**

</div>
