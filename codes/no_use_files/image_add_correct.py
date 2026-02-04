import os
import numpy
import glob
import json
S_DIR = "/home/amir-admin/projects/CNN_hand_control/datasets/val2017"
T_DIR = "/home/amir-admin/projects/CNN_hand_control/datasets/images/train"
V_DIR = "/home/amir-admin/projects/CNN_hand_control/datasets/images/val"
T_J = "/home/amir-admin/projects/CNN_hand_control/datasets/labels/train/json"
V_J = "/home/amir-admin/projects/CNN_hand_control/datasets/labels/val/json"


def dataset_correct(source_path):
    """this code is used to add the new data set to the old data set with correct names"""
    paths = sorted(glob.glob(os.path.join(source_path, "*.jpg")))
    print(f"Found {len(paths)} JPG images")
    for i, p in enumerate(paths, start=35510):
        if(i%10<3):
            os.replace(p, V_DIR + "/IMG_"+ str(i)+ ".jpg" )
            with open( V_J + "/IMG_"+ str(i)+ ".json", "w") as f:
                json.dump({
                    "class_id": 1,
                    "bbox": {
                        "center_x": 0.0,
                        "center_y": 0.0,
                        "width": 0.0,
                        "height": 0.0
                    },
                    "keypoints": {
                        "wrist": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "thumb_cmc": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "thumb_mcp": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "thumb_ip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "thumb_tip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "index_mcp": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "index_pip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "index_dip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "index_tip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "middle_mcp": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "middle_pip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "middle_dip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "middle_tip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "ring_mcp": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "ring_pip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "ring_dip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "ring_tip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "pinky_mcp": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "pinky_pip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "pinky_dip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "pinky_tip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        }
                    }
                }, f)
        else:
            with open( T_J + "/IMG_"+ str(i)+ ".json", "w") as f:
                json.dump({
                    "class_id": 1,
                    "bbox": {
                        "center_x": 0.0,
                        "center_y": 0.0,
                        "width": 0.0,
                        "height": 0.0
                    },
                    "keypoints": {
                        "wrist": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "thumb_cmc": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "thumb_mcp": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "thumb_ip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "thumb_tip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "index_mcp": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "index_pip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "index_dip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "index_tip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "middle_mcp": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "middle_pip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "middle_dip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "middle_tip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "ring_mcp": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "ring_pip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "ring_dip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "ring_tip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "pinky_mcp": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "pinky_pip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "pinky_dip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        },
                        "pinky_tip": {
                            "x": 0.0,
                            "y": 0.0,
                            "v": 0.0
                        }
                    }
                }, f)
            os.replace(p, T_DIR + "/IMG_"+ str(i)+ ".jpg" )
if __name__ == "__main__":
    dataset_correct(S_DIR)