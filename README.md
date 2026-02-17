# Word-level-lip-reader

Word-Level Lip Reading using 3D CNN (GRID Corpus)

A deep learning project that performs visual speech recognition (lip reading) using only video input.
The system predicts spoken words by analysing lip movements without using audio.


---

 Project Highlights

 Word-level lip reading
 Real-time webcam prediction
 3D Convolutional Neural Network (3D CNN)
 Training graphs + confusion matrix
 GPU training support (RTX series tested)


---

 Project Pipeline

GRID Corpus Videos
        ↓
Face Detection (dlib)
        ↓
Mouth ROI Extraction
        ↓
Frame Normalization (64×128)
        ↓
Video Tensor (C, T, H, W)
        ↓
3D CNN Model
        ↓
Word Prediction


---

 Objective

The goal of this project is to:

Detect lips from video frames

Learn temporal lip motion patterns

Classify words using a neural network

Perform live word prediction from webcam feed



---

 Dataset — GRID Corpus

The project uses the GRID audiovisual speech corpus, containing multiple speakers saying fixed-structure sentences.

Sentence structure:

command + color + preposition + letter + digit + adverb

Example:

put red at g9 now

For this project, only word-level clips were extracted.


---

 Model Architecture (3D CNN)

Why 3D CNN?

Normal CNNs learn spatial features only.
Lip reading requires spatial + temporal learning, so 3D convolutions are used.

Architecture

Conv3D → ReLU → MaxPool

Conv3D → ReLU → MaxPool

Conv3D → ReLU → MaxPool

Fully Connected layers

Dropout (regularization)


Input Shape

(Batch, Channels, Time, Height, Width)
(B, 3, 29, 64, 128)

Output Classes

bin

lay

place

set



---

⚙️ Preprocessing Pipeline

1️⃣ Detect face using dlib
2️⃣ Extract mouth landmarks (points 48–68)
3️⃣ Crop mouth ROI
4️⃣ Resize to 64×128
5️⃣ Save as .npy files

Dataset structure:

grid_word_dataset/
│
├── bin/
├── lay/
├── place/
└── set/


---

🧪 Training

Loss & Optimizer

CrossEntropyLoss

Adam optimizer


Training details

GPU: NVIDIA RTX 4050

Epochs: 20–40 recommended

Batch size: 8–16



---

📊 Results

Generated automatically:

📈 accuracy_graph.png

📉 loss_graph.png

🧩 confusion_matrix.png


Example outcome:

Training accuracy → up to ~95–100%

Real-time predictions work with webcam



---

🎥 Live Prediction

The live script:

1. Starts webcam


2. Detects lips continuously


3. Captures 29 frames when triggered


4. Runs model inference


5. Displays predicted word



Controls

S → Start word capture
Q → Quit


---

📁 Important Files

File	Purpose

grid_preprocess_word.py	Converts GRID videos to mouth ROIs
dataset_one_word.py	PyTorch dataset loader
model_one_word.py	3D CNN model
train_word_gpu.py	Training script
predict_live_word.py	Webcam live prediction
word_model_gpu.pth	Trained weights
shape_predictor_68_face_landmarks.dat	Facial landmark model



---

 Installation

1️⃣ Create environment

python -m venv .venv

Activate (PowerShell):

.\.venv\Scripts\Activate


---

2️⃣ Install dependencies

pip install torch torchvision torchaudio
pip install opencv-python dlib imutils pynput numpy matplotlib


---

3️⃣ Train model

python train_word_gpu.py


---

4️⃣ Run live prediction

python predict_live_word.py


---

⚠️ Notes

shape_predictor_68_face_landmarks.dat is ~97MB and not uploaded.

Download separately from the dlib model repository.

GPU strongly recommended for training.



---

🔬 Future Improvements

Add full GRID vocabulary (colors, numbers, letters)

Temporal smoothing for stable live predictions

Replace 3D CNN with Transformer-based architecture

Use speaker-independent evaluation


---
 Acknowledgements

GRID Corpus dataset

dlib facial landmark detector

PyTorch deep learning framework
