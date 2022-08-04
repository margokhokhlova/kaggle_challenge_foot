import os
import glob
import cv2
import pandas as pd

from utils import viz

if __name__ == "__main__":
    root_path = "nfl-health-and-safety-helmet-assignment"
    image_labels_file = "image_labels.csv"
    train_labels_file = "train_labels.csv"
    baseline_helmets_file = "train_baseline_helmets.csv"
    image_dir = "images"
    train_videos = "train"
    test_videos = "test"
    image_labels = pd.read_csv(f"{root_path}/{image_labels_file}")  #TODO remove pandas here
    image_files = glob.glob(f"{root_path}/{image_dir}/*")
    print(f"Total number of images: {len(image_files)}, total number of labels is {len(image_labels)}, the unique labels are {image_labels['label'].unique().tolist()}\n")
    # visualize helmets on an image
    image_selected = image_files[0]
    image_name = os.path.basename(image_selected)
    image = cv2.imread(image_selected)
    selected_labels = image_labels[image_labels['image'] == image_name]
    bboxes = selected_labels.iloc[:, 2:].to_numpy()
    labels = selected_labels.iloc[:, 1].to_numpy()
    image_viz = viz.draw_bboxes(image.copy(), bboxes, labels)
    # cv2.imshow('Image with helmets', image_viz)
    # cv2.waitKey(1000)
    # check the videos
    video_files = os.listdir(f"{root_path}/{train_videos}")
    train_labels_df = pd.read_csv(f"{root_path}/{train_labels_file}")
    baseline_bbx_df = pd.read_csv(f"{root_path}/{baseline_helmets_file}")
    videofile_name = video_files[1]
    # add data to privided baseline bboxes 
    baseline_bbx_df["video"] = baseline_bbx_df["video_frame"].str.split("_").str[:3].str.join("_")
    baseline_bbx_df["frame"] = baseline_bbx_df["video_frame"].str.split("_").str[-1].astype("int")
    # and the GT bboxes
    train_labels_df["video"] = train_labels_df["video_frame"].str.split("_").str[:3].str.join("_")+'.mp4'
    train_labels_df["frame"] = train_labels_df["video_frame"].str.split("_").str[-1].astype("int")
    
    print(train_labels_df.head())
    print(videofile_name)
    # play a video
    videocap = cv2.VideoCapture(os.path.join(f"{root_path}/{train_videos}",videofile_name))
    fps = videocap.get(cv2.CAP_PROP_FPS)
    frame = 0
    while True:
        it_worked, img = videocap.read()
        if not it_worked:
            break
        # We need to add 1 to the frame count to match the label frame index
        # that starts at 1
        frame += 1

        # Let's add a frame index to the video so we can track where we are
        img_name = f"{videofile_name}_frame{frame}"
        cv2.putText(img, img_name, (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0,0,255), thickness=2)

        # Now, add the boxes
        boxes = train_labels_df.query("video == @videofile_name and frame == @frame")
        if len(boxes) == 0:
            print("Boxes incorrect")
            continue
        for box in boxes.itertuples(index=False):
            # Filter for definitive head impacts and turn red
            if box.isDefinitiveImpact == True:
                color, thickness = (0,0,255), 3
            else:
                color, thickness = (255,0,0), 1
            cv2.rectangle(img, (box.left, box.top), (box.left+box.width, box.top+box.height),
                          color, thickness=thickness)
            cv2.putText(img, box.label, (box.left+1, max(0, box.top-20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)
        cv2.imshow('video', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
    videocap.release()      
    cv2.destroyAllWindows()

      


