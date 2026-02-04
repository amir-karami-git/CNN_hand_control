import os
import glob
import cv2

IMG_DIR = "/home/amir-admin/projects/CNN_hand_control/datasets/val2017"  # <-- change this
TARGET = 224

def crop_folder_overwrite(img_dir):
    print(img_dir)
    paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"Found {len(paths)} JPG images")

    for i, p in enumerate(paths, start=1):
        img = cv2.imread(p)
        if img is None:
            print(f"[{i}] Failed reading {p}")
            continue

        out = cv2.resize(img,(TARGET,TARGET))
        cv2.imwrite(p, out)

        if i % 200 == 0:
            print(f"Processed {i}/{len(paths)}")

    print("Finished cropping all images.")
if __name__ == "__main__":
    crop_folder_overwrite(IMG_DIR)