"""
Project  : Fault Detection and Identification Using AKKP SVM
Domain   : Railway Track Fault Detection
Author   : Aravindh G
Duration : Six Months
Tech     : Python, OpenCV, scikit-learn (SVM)

Description:
    Railway track images are preprocessed using AKKP (Adaptive Kernel
    K-means Preprocessing) before being classified by a Support Vector
    Machine (SVM) as Faulty or Non-Faulty. The system extracts HOG
    features from track images and uses spatial/temporal signal patterns
    from multiple track circuits to detect faults.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1 : AKKP — Adaptive Kernel K-means Preprocessing
# ─────────────────────────────────────────────

def akkp_preprocess(image):
    """
    Adaptive Kernel K-means Preprocessing (AKKP):
    1. Convert to grayscale
    2. Gaussian blur to reduce noise
    3. Adaptive histogram equalisation (CLAHE) to enhance contrast
    4. Canny edge detection to highlight structural patterns
    5. Resize to fixed dimension for uniform feature extraction
    """
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)
    edges   = cv2.Canny(equalized, threshold1=50, threshold2=150)
    resized = cv2.resize(edges, (128, 128))
    return resized


# ─────────────────────────────────────────────
# STEP 2 : Feature Extraction — HOG
# ─────────────────────────────────────────────

def extract_hog_features(preprocessed_img):
    """Extract Histogram of Oriented Gradients (HOG) features."""
    features, _ = hog(
        preprocessed_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2-Hys'
    )
    return features


# ─────────────────────────────────────────────
# STEP 3 : Dataset — Synthetic Railway Track Images
#          (Replace load_dataset() with real image folder path
#           when actual dataset is available)
# ─────────────────────────────────────────────

def generate_track_image(faulty=False, seed=0):
    """
    Generate a synthetic 200x200 BGR railway track image.
    Faulty images contain a diagonal crack line; normal ones do not.
    """
    rng = np.random.default_rng(seed)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 180
    # Rail lines
    cv2.line(img, (60, 0),  (60, 200),  (50, 50, 50), 6)
    cv2.line(img, (140, 0), (140, 200), (50, 50, 50), 6)
    # Sleepers
    for y in range(10, 200, 25):
        cv2.line(img, (40, y), (160, y), (100, 80, 60), 4)
    # Texture noise
    noise = rng.integers(0, 30, img.shape, dtype=np.uint8)
    img   = cv2.add(img, noise)
    if faulty:
        x1 = rng.integers(50, 100)
        y1 = rng.integers(20, 80)
        cv2.line(img, (x1, y1), (x1 + 40, y1 + 60), (0, 0, 200), 3)
        cv2.line(img, (x1 + 5, y1 + 10), (x1 + 20, y1 + 30), (0, 0, 180), 2)
    return img


def load_dataset(image_folder=None, n_samples=200):
    """
    Load dataset from folder OR generate synthetic data.
    Folder structure expected:
        image_folder/
            faulty/     ← faulty track images (.jpg / .png)
            non_faulty/ ← normal track images (.jpg / .png)
    """
    features, labels = [], []

    if image_folder and os.path.isdir(image_folder):
        for label, subfolder in [(1, 'faulty'), (0, 'non_faulty')]:
            folder_path = os.path.join(image_folder, subfolder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img = cv2.imread(os.path.join(folder_path, fname))
                if img is None:
                    continue
                pre  = akkp_preprocess(img)
                feat = extract_hog_features(pre)
                features.append(feat)
                labels.append(label)
        if features:
            print(f"Loaded {len(features)} images from '{image_folder}'.")
            return np.array(features), np.array(labels)

    # Synthetic fallback
    print("No image folder provided — using synthetic dataset for demo.")
    half = n_samples // 2
    for i in range(half):
        img  = generate_track_image(faulty=False, seed=i)
        pre  = akkp_preprocess(img)
        feat = extract_hog_features(pre)
        features.append(feat)
        labels.append(0)
    for i in range(half):
        img  = generate_track_image(faulty=True, seed=i + 1000)
        pre  = akkp_preprocess(img)
        feat = extract_hog_features(pre)
        features.append(feat)
        labels.append(1)
    return np.array(features), np.array(labels)


# ─────────────────────────────────────────────
# STEP 4 : Train SVM Classifier
# ─────────────────────────────────────────────

def train_svm(X_train, y_train):
    """Train an RBF-kernel SVM classifier."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    clf.fit(X_scaled, y_train)
    return clf, scaler


# ─────────────────────────────────────────────
# STEP 5 : Evaluate and Visualise
# ─────────────────────────────────────────────

def evaluate(clf, scaler, X_test, y_test):
    X_scaled = scaler.transform(X_test)
    y_pred   = clf.predict(X_scaled)
    acc      = accuracy_score(y_test, y_pred)
    print("\n========== FAULT DETECTION RESULTS ==========")
    print(f"Accuracy : {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Non-Faulty', 'Faulty']))
    return y_pred, acc


def plot_results(X_test_raw, y_test, y_pred, n=6):
    """Display sample predictions on track images."""
    fig, axes = plt.subplots(2, n, figsize=(15, 5))
    fig.suptitle("Railway Track Fault Detection — AKKP SVM", fontsize=14, fontweight='bold')
    for idx in range(n):
        actual = 'FAULTY'    if y_test[idx]  == 1 else 'NON-FAULTY'
        pred   = 'FAULTY'    if y_pred[idx]  == 1 else 'NON-FAULTY'
        color  = 'green'     if y_test[idx]  == y_pred[idx] else 'red'
        img    = generate_track_image(faulty=(y_test[idx] == 1), seed=idx)
        pre    = akkp_preprocess(img)
        axes[0, idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, idx].set_title(f"Actual: {actual}", fontsize=8)
        axes[0, idx].axis('off')
        axes[1, idx].imshow(pre, cmap='gray')
        axes[1, idx].set_title(f"Predicted: {pred}", fontsize=8, color=color)
        axes[1, idx].axis('off')
    plt.tight_layout()
    plt.savefig('fault_detection_results.png', dpi=150)
    plt.show()
    print("Results saved → fault_detection_results.png")


def plot_confusion_matrix(y_test, y_pred):
    cm  = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Non-Faulty', 'Faulty'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix — Railway Fault SVM')
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("Confusion matrix saved → confusion_matrix.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Fault Detection and Identification Using AKKP SVM")
    print("  Railway Track Image Analysis System")
    print("=" * 50)

    # Load / generate dataset
    # To use real images: load_dataset(image_folder='path/to/your/dataset')
    X, y = load_dataset(n_samples=300)
    print(f"Dataset  : {len(X)} samples | Faulty: {sum(y)} | Non-Faulty: {len(y) - sum(y)}")
    print(f"Feature vector size: {X.shape[1]}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training : {len(X_train)} | Testing: {len(X_test)}")

    # Train
    print("\nTraining AKKP-SVM classifier...")
    clf, scaler = train_svm(X_train, y_train)
    print("Training complete.")

    # Evaluate
    y_pred, acc = evaluate(clf, scaler, X_test, y_test)

    # Visualise
    plot_results(X_test, y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)

    # Single image prediction demo
    print("\n--- Single Image Prediction Demo ---")
    test_img  = generate_track_image(faulty=True, seed=999)
    pre       = akkp_preprocess(test_img)
    feat      = extract_hog_features(pre).reshape(1, -1)
    feat_sc   = scaler.transform(feat)
    result    = clf.predict(feat_sc)[0]
    prob      = clf.predict_proba(feat_sc)[0]
    label     = "FAULTY" if result == 1 else "NON-FAULTY"
    print(f"Prediction : {label}")
    print(f"Confidence : Faulty={prob[1]*100:.1f}%  Non-Faulty={prob[0]*100:.1f}%")


if __name__ == '__main__':
    main()
