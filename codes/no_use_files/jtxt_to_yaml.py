import os
import glob
import json

LABEL_DIR = "/home/amir-admin/projects/CNN_hand_control/datasets/labels/train/text"
JSON_DIR  = "/home/amir-admin/projects/CNN_hand_control/datasets/labels/train/json"

# create output directory if it doesn't exist
os.makedirs(JSON_DIR, exist_ok=True)

file_paths = sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt")))

for i, txt_path in enumerate(file_paths, start=1):
    base = os.path.splitext(os.path.basename(txt_path))[0]   # e.g. "000123"
    json_path = os.path.join(JSON_DIR, base + ".json")

    annotations = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            nums = list(map(float, line.split()))

            # safety check
            if len(nums) != 68:
                print(f"Skipping malformed label in {txt_path}")
                continue

            label_dict = {
                "class_id": int(nums[0]),

                "bbox": {
                    "center_x": nums[1],
                    "center_y": nums[2],
                    "width": nums[3],
                    "height": nums[4],
                },

                "keypoints": {
                    "wrist":        {"x": nums[5],  "y": nums[6],  "v": nums[7]},
                    "thumb_cmc":    {"x": nums[8],  "y": nums[9],  "v": nums[10]},
                    "thumb_mcp":    {"x": nums[11], "y": nums[12], "v": nums[13]},
                    "thumb_ip":     {"x": nums[14], "y": nums[15], "v": nums[16]},
                    "thumb_tip":    {"x": nums[17], "y": nums[18], "v": nums[19]},

                    "index_mcp":    {"x": nums[20], "y": nums[21], "v": nums[22]},
                    "index_pip":    {"x": nums[23], "y": nums[24], "v": nums[25]},
                    "index_dip":    {"x": nums[26], "y": nums[27], "v": nums[28]},
                    "index_tip":    {"x": nums[29], "y": nums[30], "v": nums[31]},

                    "middle_mcp":   {"x": nums[32], "y": nums[33], "v": nums[34]},
                    "middle_pip":   {"x": nums[35], "y": nums[36], "v": nums[37]},
                    "middle_dip":   {"x": nums[38], "y": nums[39], "v": nums[40]},
                    "middle_tip":   {"x": nums[41], "y": nums[42], "v": nums[43]},

                    "ring_mcp":     {"x": nums[44], "y": nums[45], "v": nums[46]},
                    "ring_pip":     {"x": nums[47], "y": nums[48], "v": nums[49]},
                    "ring_dip":     {"x": nums[50], "y": nums[51], "v": nums[52]},
                    "ring_tip":     {"x": nums[53], "y": nums[54], "v": nums[55]},

                    "pinky_mcp":    {"x": nums[56], "y": nums[57], "v": nums[58]},
                    "pinky_pip":    {"x": nums[59], "y": nums[60], "v": nums[61]},
                    "pinky_dip":    {"x": nums[62], "y": nums[63], "v": nums[64]},
                    "pinky_tip":    {"x": nums[65], "y": nums[66], "v": nums[67]},
                }
            }

            annotations.append(label_dict)

    # if there is exactly ONE hand per image, save the single dict
    output = annotations[0] if len(annotations) == 1 else annotations

    with open(json_path, "w") as jf:
        json.dump(output, jf, indent=4)

    print(f"[{i}/{len(file_paths)}] Saved {json_path}")
