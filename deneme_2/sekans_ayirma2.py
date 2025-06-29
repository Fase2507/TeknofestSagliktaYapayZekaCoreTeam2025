import os, shutil, cv2, pydicom
import numpy as np
import pandas as pd
from glob import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
# 1️⃣  Gather DICOM paths
dicom_paths = glob(
    "../../../Yarışma 2.aşama MR Veri Seti Kümesi/Yarışma 2.aşama veri seti kümesi/Vaka_*/*/*/*.dcm",
    recursive=True
)


# 2️⃣  Feature extraction helpers
def extract_features(img):
    img = img.astype(np.float32)
    img = cv2.resize(img, (224, 224))
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    mean  = img.mean()
    std   = img.std()

    edges = cv2.Canny((img * 255).astype(np.uint8), 50, 150)
    edge_density = (edges > 0).mean()

    csf_ratio  = (img > 0.80).mean()   # very bright pixels
    dark_ratio = (img < 0.10).mean()   # very dark pixels

    return [mean, std, edge_density, csf_ratio, dark_ratio]

# 3️⃣  Extract features for every DICOM
features, filenames = [], []
for path in tqdm(dicom_paths, desc="Reading DICOMs"):
    try:
        ds  = pydicom.dcmread(path)
        img = ds.pixel_array
        if img.ndim > 2:      # multiframe
            img = img[0]
        features.append(extract_features(img))
        filenames.append(path)
    except Exception as e:
        print(f"[WARN] Could not process {path}: {e}")

features = np.asarray(features)
scaler=StandardScaler()
scaler.fit(features)
scaled   = StandardScaler().fit_transform(features)
# 4️⃣  K‑means clustering (k=3)
kmeans  = KMeans(n_clusters=3, random_state=42)
labels  = kmeans.fit_predict(scaled)

# 5️⃣  Build DataFrame
cols   = ["mean", "std", "edge_density", "csf_ratio", "dark_ratio"]
df     = pd.DataFrame(features, columns=cols)
df["file"]    = filenames
df["cluster"] = labels

# 6️⃣  Quick diagnostics
print("\n=== Cluster sizes ===")
print(df["cluster"].value_counts(), "\n")

# 7️⃣  Map cluster ➜ sequence **(adjust after inspection!)**
cluster_to_name = {0: "T2A", 1: "DWI", 2: "ADC"}  # <- change if needed
df["label"] = df["cluster"].map(cluster_to_name)

# 8️⃣  Move / copy files into folders
for _, row in df.iterrows():
    label_dir = os.path.join("classified", row["label"])
    os.makedirs(label_dir, exist_ok=True)

    src = row["file"]
    dst = os.path.join(label_dir, os.path.basename(src))

    # If you worry about duplicate basenames, uncomment next two lines
    # uid = os.path.splitext(os.path.basename(src))[0]
    # dst = os.path.join(label_dir, f"{uid}_{row['cluster']}.dcm")

    shutil.copy(src, dst)
numeric_cols = ["mean", "std", "edge_density", "csf_ratio", "dark_ratio"]
df[numeric_cols] = df[numeric_cols].round(3)
# 9️⃣  Save CSV with human‑readable labels
df.drop(columns=["cluster"]).to_csv("dicom_clustered.csv", index=False)
print("✅  Finished. CSV saved as dicom_clustered.csv")

# The StandardScaler used to normalize feature vectors
# The trained KMeans model
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")


