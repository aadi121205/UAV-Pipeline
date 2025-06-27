import os, glob, cv2, numpy as np, gc
from datetime import datetime
from tqdm import tqdm

# === CONFIGURATION ===
FEATURE_TYPE = "sift"  # Choose "sift" or "orb"
IMG_FOLDER = os.path.expanduser("~/Desktop/Mapping data (Copy)")
SAVE_DIR = os.path.expanduser("~/Desktop/stitchedimage/")
RESIZE_TO = (1000, 750)  # Resize all images to this fixed size
REPROJ_THRESH = 4.0
NUM_FEATURES = 10000
MIN_MATCHES = 10  # Require this many good matches to proceed

# === SETUP DIRECTORIES ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === FEATURE EXTRACTOR SETUP ===
cv2.setNumThreads(4)
cv2.ocl.setUseOpenCL(False)

def get_detector(ftype):
    return cv2.SIFT_create(NUM_FEATURES) if ftype == "sift" else cv2.ORB_create(NUM_FEATURES)

def stitch(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_trans = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_trans), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    trans = [-xmin, -ymin]
    Ht = np.array([[1, 0, trans[0]], [0, 1, trans[1]], [0, 0, 1]])
    result = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    result[trans[1]:h1+trans[1], trans[0]:w1+trans[0]] = img1
    return result

def resize_to_fixed(img):
    return cv2.resize(img, RESIZE_TO)

def build_mosaic(image_paths, feature_type):
    detector = get_detector(feature_type)
    matcher = (cv2.BFMatcher(cv2.NORM_HAMMING) if feature_type == "orb"
               else cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50)))

    stitched = None
    for idx, path in enumerate(tqdm(image_paths, desc="Stitching", ncols=100)):
        img = cv2.imread(path)
        if img is None:
            continue
        img = resize_to_fixed(img)
        if stitched is None:
            stitched = img
            continue
        kp1, des1 = detector.detectAndCompute(cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = detector.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            continue
        matches = matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < MIN_MATCHES:
            continue
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good])
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, REPROJ_THRESH)
        if H is None:
            continue
        try:
            stitched = stitch(stitched, img, H)
        except:
            continue
        gc.collect()

    out_path = os.path.join(SAVE_DIR, f"city_imgs-final.png")
    if stitched is not None:
        cv2.imwrite(out_path, stitched)
        print(f"\n✅ Final mosaic saved: {out_path}")
    else:
        print("\n❌ Stitching failed.")

# === EXECUTE ===
if __name__ == "__main__":
    all_paths = sorted(glob.glob(os.path.join(IMG_FOLDER, "*.jpg")) + glob.glob(os.path.join(IMG_FOLDER, "*.JPG")))[213:220]
    if len(all_paths) < 2:
        print("[ERROR] Not enough images.")
    else:
        build_mosaic(all_paths, FEATURE_TYPE)