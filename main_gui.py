import os
import pyrealsense2 as rs
import numpy as np
import cv2
import csv
from tkinter import *
from OpenGL.GLUT import *
import viewer as gl


class storing_data():
    def __init__(self, base_root):
        
        self.imgages = os.path.join(base_root, 'images')
        self.point_cloud = os.path.join(base_root, 'point_clouds')
        self.recap_root = os.path.join(base_root, 'recap')

class StreamRealsense:
    def __init__(self, w=1280, h=720, fps=30, streaming = True):

        self.pipeline = rs.pipeline()
        self.streaming = streaming # True or False, if True, the video stream will be displayed otherwise not

        # Create a config object
        self.config = rs.config()

        # Enable color and depth streams
        self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def start(self):
        # Start the pipeline
        self.pipeline.start(self.config)

    def stop(self):
        # Stop the pipeline
        self.pipeline.stop()
    
    def set_exposure(self, exposure_value):
        device = self.pipeline.get_active_profile().get_device()
        depth_sensor = device.first_depth_sensor()
        depth_sensor.set_option(rs.option.exposure, exposure_value)

    def get_frame(self):

        frames = self.pipeline.wait_for_frames(10000)    # Wait for a coherent pair of frames: depth and color
        aligned_frames = self.align.process(frames) # Align the depth frame to color frame

        # Get color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics # Transformation matrix that map 2d points into 3d world coordinates

        color_image = np.asanyarray(color_frame.get_data()) # color frames
        depth_image = np.asanyarray(depth_frame.get_data()) # depth images, enbed the depth into the color of the pixel

        return color_image, depth_image, depth_frame, depth_intrin
    
def file_writer(writer_input, win):

    """
    Generate a csv file containing information about the retrieved data. Write the point cloud to a file and save the images.

    Args:
        root (storing_data): The root directory to store the data
        idx (int): The index of the point cloud
        timestamp (int): The timestamp of the point cloud
        point_cloud_to_save (pyzed.sl.Mat): The point cloud to save
        image_to_save (pyzed.sl.Mat, optional): The left image to save. Default to None.

    Returns:
        err (pyzed.sl.ERROR_CODE): The error code if the point cloud was not saved
        idx (int): The index of the point cloud
    """
    
    win.destroy()
    
    root, idx, timestamp, dist, point_cloud, image = writer_input

    name = "berry_"+str(idx)
                
    err = point_cloud.write(os.path.join(root.point_cloud,name+'.ply'))
    if image is not None: cv2.imwrite(os.path.join(root.imgages,name+'.jpg'), image.get_data())
       

    with open(os.path.join(root.recap_root, 'list.csv'), 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow([name, timestamp, round(dist,4), err,'SUCCESS' if image is not None else 'ERROR'])
        f.close()

def displayer(camera, viewer, point_cloud, image):

    """Display the data stream (let and right image, depth point cloud) from the ZED camera

    Args:
        zed (pyzed.sl.Camera): The ZED camera
        viewer (pyzed.gl.GLViewer): The viewer to display the point cloud as Gl object
        image (pyzed.sl.Mat): The left image to display
        point_cloud (pyzed.sl.Mat): The point cloud to display

    Returns:
        None
    """

    # Retrieve resolution from the screen where the data stream is displayed
    sc_w = int(glutGet(GLUT_SCREEN_WIDTH))
    sc_h = int(glutGet(GLUT_SCREEN_HEIGHT))

    # Retrieve the left image, right image, and point cloud
    camera.retrieve_image(image, rs.stream.color)
    camera.retrieve_image(point_cloud, rs.stream.depth)

    # Display the left image
    cv2.namedWindow("LEFT_IMAGES", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LEFT_IMAGES", int(sc_w*0.24), int(sc_h*0.4))
    cv2.moveWindow("LEFT_IMAGES", int(sc_w*0.52), int(sc_h*0.55))
    cv2.imshow("LEFT_IMAGES", image)
    cv2.waitKey(1)

    # Display the right image
    cv2.namedWindow("RIGHT_IMAGES", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RIGHT_IMAGES", int(sc_w*0.24), int(sc_h*0.4))
    cv2.moveWindow("RIGHT_IMAGES", int(sc_w*0.85), int(sc_h*0.55))
    cv2.imshow("RIGHT_IMAGES", point_cloud)
    cv2.waitKey(1)

    viewer.updateData(point_cloud)

def compute_centre(res, point_cloud):
    
        """
        Compute the centre of the point cloud
    
        Args:
            res (tuple): The resolution of the point cloud
            point_cloud (pyzed.sl.Mat): The point cloud to compute the centre
    
        Returns:
            distance (float): The distance of the centre of the point cloud
        """
    
        # Retrieve the point cloud data
        point_cloud_data = point_cloud.get_data()
    
        # Compute the centre of the point cloud
        centre = point_cloud_data[int(res[0]/2), int(res[1]/2)]
    
        # Compute the distance of the centre of the point cloud
        distance = np.sqrt(centre[0]**2 + centre[1]**2 + centre[2]**2)
    
        return distance


def main():

    """
    Main function to run the acquisition GUI
    """
    
    main_root = '/test_save/'
    dataset_folders = storing_data(main_root)

    camera = StreamRealsense()
    camera.start()

    context = rs.context()
    device = context.query_devices()
    camera_model = device.get_info(rs.camera_info.name)

    res = (1280, 720)

    #viewer = gl.GLViewer()
    #viewer.init(1 , sys.argv,  camera_model, res)


    viewer = gl.GLViewer()
    viewer.init(1, sys.argv, "RealSense Camera", (1280, 720))


    if 'list.csv' not in os.listdir(dataset_folders.recap_root):
        with open(os.path.join(dataset_folders.recap_root, 'list.csv'), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(["filename","timestamp", "centre_depth", "point_cloud", "image", "index"])
            f.close()
    else:
        with open(os.path.join(dataset_folders.recap_root, 'list.csv'), 'r', encoding='UTF8') as f:
            final_line = f.readlines()[-1]
            previous = final_line.split(',')[-1]
            f.close()
            if previous == 'index\n':
                i = 0
            else:
                i = int(previous) + 1

    while viewer.is_available():
        color_image, depth_image, depth_frame, depth_intrin = camera.get_frame()

        # Display the images
        displayer(color_image, depth_image, viewer)

        if viewer.save_data:
            dist = compute_centre((1280, 720), depth_image)

            # Assuming the point cloud is in depth_image format
            point_cloud_to_save = depth_image
            image_to_save = color_image

            timestamp = rs.frame.get_timestamp(depth_frame)

            writer_input = [dataset_folders, i, timestamp, dist, point_cloud_to_save, image_to_save]

            file_writer(writer_input, viewer)

            i += 1
            viewer.save_data = False

    cv2.destroyAllWindows()
    viewer.exit()

if __name__ == "__main__":
    main()