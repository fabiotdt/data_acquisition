import cv2
import pyrealsense2 as rs
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import os
import csv
import datetime
import pandas as pd

class storing_data():
    def __init__(self, base_root='test_save/'):
        
        self.colour_images = os.path.join(base_root, 'images')
        self.depth_images = os.path.join(base_root, 'depth_images')
        self.point_clouds = os.path.join(base_root, 'point_clouds')
        self.dataset = os.path.join(base_root, 'dataset')
        
        os.makedirs(self.colour_images, exist_ok=True)
        os.makedirs(self.depth_images, exist_ok=True)
        os.makedirs(self.point_clouds, exist_ok=True)
        os.makedirs(self.dataset, exist_ok=True)

class ImageApp:
    def __init__(self, root_frame, root_path, w=1280, h=720, fps=30, streaming=True):
        
        self.root_frame = root_frame
        self.root_path = root_path
        self.pipeline = rs.pipeline()
        self.streaming = streaming  # True or False, if True, the video stream will be displayed otherwise not

        self.res = (w, h)

        # Create a config object
        self.config = rs.config()

        # Enable color and depth streams
        self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Create a video frame on the right
        self.video_frame = Frame(self.root_frame)
        self.video_frame.grid(row=6, column=3, rowspan=15, columnspan=3, padx=10, pady=20)
        self.video_label = Label(self.video_frame)
        self.video_label.pack()

        self.depth_frame_display = Frame(self.root_frame)
        self.depth_frame_display.grid(row=6, column=6, rowspan=15, columnspan=3, padx=10, pady=20)
        self.depth_label_display = Label(self.depth_frame_display)
        self.depth_label_display.pack()

        self.start()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames(10000)  # Wait for a coherent pair of frames: depth and color
        aligned_frames = self.align.process(frames)  # Align the depth frame to color frame

        # Get color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())  # color frames
        depth_image = np.asanyarray(depth_frame.get_data())  # depth images, embed the depth into the color of the pixel

        return color_image, depth_image, depth_frame

    def start(self):
        # Start the pipeline
        self.pipeline.start(self.config)
        self.update_video_stream()

    def stop(self):
        # Stop the pipeline
        self.pipeline.stop()

    def depth_to_cloud(self, depth_frame):
        points = rs.pointcloud()
        points.map_to(depth_frame)
        cloud = points.calculate(depth_frame)

        return cloud

    def update_video_stream(self):
        color_image, depth_image, _ = self.get_frame()

        prop = self.res[0] / self.res[1]

        new_w = int(self.root_frame.winfo_screenwidth() / 4.5)
        new_h = int(new_w * prop)

        image_display = cv2.resize(color_image, (new_h, new_w))
        image_display = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)

        photo = ImageTk.PhotoImage(image=Image.fromarray(image_display))
        self.video_label.config(image=photo)
        self.video_label.image = photo

        depth_display = cv2.resize(depth_image, (new_h, new_w))
        depth_display = cv2.applyColorMap(cv2.convertScaleAbs(depth_display, alpha=0.03), cv2.COLORMAP_JET)

        photo_depth = ImageTk.PhotoImage(image=Image.fromarray(depth_display))
        self.depth_label_display.config(image=photo_depth)
        self.depth_label_display.image = photo_depth

        #print('Updating video stream')
        self.root_frame.after(20, self.update_video_stream)

    def submit_action(self, name):
        
        color_image, depth_image, depth_frame = self.get_frame()

        # Save the images
        cv2.imwrite(os.path.join(self.root_path.colour_images, name + '.jpg'), color_image)
        cv2.imwrite(os.path.join(self.root_path.depth_images, name + '.jpg'), depth_image)
        # Save the depth point cloud
        #point_cloud = self.depth_to_cloud(depth_frame)
        #point_cloud.export_to_ply(os.path.join(self.root_path.point_clouds, name + '.ply'), depth_frame)

        #self.update_video_stream()

class FileCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("File Counter App")
        self.folder_path = storing_data() 
        self.berry_counter_var = StringVar()
        self.create_widgets()

    def create_widgets(self):
        berry_txt = Label(self.root, text="Number of images: ", font=('calibre', 10))
        berry_txt.grid(row=20, column=0, padx=1, pady=5)
        berry_in = Entry(self.root, textvariable=self.berry_counter_var, state="readonly", font=('calibre', 10))
        berry_in.grid(row=21, column=1, padx=1, pady=5)
        self.update_file_counts()

    def update_file_counts(self):
        dataset_path = os.path.join(self.folder_path.dataset, 'dataset.csv')
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            berry_count = df['index'].count()
        else:
            berry_count = 0
        self.berry_counter_var.set(f"{berry_count}")
        self.root.after(1000, self.update_file_counts)

def save_data(input, image_app):
    """
    This function will save the image, a csv containing data collection information, and a csv containing measurement information.
    """
    root = storing_data()

    # Create dataset.csv if it doesn't exist
    dataset_path = os.path.join(root.dataset, 'dataset.csv')
    if not os.path.exists(dataset_path):
        os.makedirs(root.dataset, exist_ok=True)
        with open(dataset_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            #writer.writerow(["filename", "timestamp", "colour_image", "depth_image", "point_cloud", "index"])
            writer.writerow(["filename", "timestamp", "colour_image", "depth_image", "index"])

    # Determine the next index
    with open(dataset_path, 'r', encoding='UTF8') as f:
        reader = csv.reader(f)
        lines = list(reader)
        if len(lines) <= 1:  # Only header exists
            i = 0
        else:
            final_line = lines[-1]
            previous = final_line[-1].strip()
            print(f'this is the previous: {previous}')
            try:
                i = int(previous) + 1
            except ValueError:
                i = 0  # In case of any unexpected value, reset to 0
        f.close()

    name = input + '_' + str(i)

    ct = datetime.datetime.now()
    timestamp = ct.timestamp()
    image_app.submit_action(name)

    # Append new entry to the dataset
    with open(dataset_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name,
                         timestamp,
                         os.path.join(root.colour_images, name + '.jpg'),
                         os.path.join(root.depth_images, name + '.jpg'),
                         #os.path.join(root.point_clouds, name + '.ply'),
                         i])
        f.close()

def main():
    win = Tk()
    win.title("Berry Image Manager")
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    win.geometry(f"{screen_width}x{screen_height}")
    win['padx'] = 20
    win['pady'] = 20
    
    image_app = ImageApp(win, storing_data())
    FileCounterApp(win)  

    sub_btn = Button(win, text='Submit', command=lambda: save_data("berry", image_app), font=('calibre', 20, 'bold'))
    sub_btn.grid(row=15, column=0, columnspan=3, padx=1, pady=20)

    realsense_image = Label(win, text="RealSense image", font=('calibre', 20, 'bold'))
    realsense_image.grid(row=5, column=3, columnspan=3, padx=1, pady=10)

    win.mainloop()

if __name__ == '__main__':
    main()

