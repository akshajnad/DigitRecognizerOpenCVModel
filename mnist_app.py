import os
import subprocess
import zipfile

import numpy as np
import pandas as pd
import cv2

from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression

###############################################################################
# 1) FLASK APP SETUP
###############################################################################
app = Flask(__name__)
app.config["SECRET_KEY"] = "some_secret_key"

# Kaggle competition name
KAGGLE_COMPETITION = "digit-recognizer"
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"  # Not strictly needed, but listed for reference

###############################################################################
# 2) DOWNLOAD DATA FROM KAGGLE IF train.csv NOT FOUND
###############################################################################
if not os.path.exists(TRAIN_CSV):
    print(f"'{TRAIN_CSV}' not found. Downloading from Kaggle...")

    # Run Kaggle CLI to download the competition files (train.csv, test.csv).
    # The "--force" ensures redownload if something is partial/corrupt.
    subprocess.run([
        "kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION,
        "--force", "-p", "."
    ], check=True)

    # The downloaded files are in ZIP format: 'train.csv.zip', 'test.csv.zip', etc.
    # Unzip any .zip files in the current directory.
    for file in os.listdir("."):
        if file.endswith(".zip"):
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(".")
            os.remove(file)  # optionally remove the .zip after extraction

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Failed to find '{TRAIN_CSV}' even after download. Check Kaggle credentials."
        )
else:
    print(f"'{TRAIN_CSV}' found locally. Skipping Kaggle download.")

###############################################################################
# 3) LOAD A PORTION OF MNIST (train.csv)
#    The file has ~42,000 rows. We'll load a subset for speed.
###############################################################################
df = pd.read_csv(TRAIN_CSV)
# df looks like:
#   label, pixel0, pixel1, ..., pixel783
# We'll keep ~5000 rows for demonstration (adjust as desired).
df = df.head(5000)  # reduce for quicker training

# Separate features (X) and labels (y)
y = df["label"].values
X = df.drop("label", axis=1).values  # shape: (5000, 784)

###############################################################################
# 4) TRAIN A SIMPLE CLASSIFIER (Logistic Regression)
###############################################################################
clf = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,  # enough iterations for convergence
    multi_class="auto"
)
clf.fit(X, y)
print("Training complete. Classes learned:", clf.classes_)

###############################################################################
# 5) FLASK ROUTE FOR UPLOADING AN IMAGE & CLASSIFYING
###############################################################################
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 1) Retrieve the uploaded file
        file = request.files.get("image_file")
        if not file:
            return render_template("index.html", prediction="No file uploaded.")

        # 2) Convert file to a CV2 image (grayscale)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # read in grayscale

        if img is None:
            return render_template("index.html", prediction="Invalid image file.")

        # 3) Resize to 28x28 (same as MNIST), flatten
        img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        # MNIST is 28x28 grayscale, each pixel is in range 0..255
        # Optionally invert or normalize if your input is dark-on-white
        # But let's assume white digit on black background for best results
        features = img_resized.reshape(1, -1).astype(np.float32)

        # 4) Predict with logistic regression
        pred_label = clf.predict(features)[0]

        return render_template(
            "index.html",
            prediction=f"Predicted Digit: {pred_label}"
        )

    # For GET request, just display the upload form
    return render_template("index.html", prediction=None)

###############################################################################
# 6) RUN THE APP
###############################################################################
if __name__ == "__main__":
    app.run(debug=True)
