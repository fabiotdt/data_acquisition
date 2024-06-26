import os 

class storing_data():
    def __init__(self, base_root):
        #self.base_root = '/media/jetson/Volume/Data'
        
        self.left_img_root = os.path.join(base_root, 'left_image')
        self.right_img_root = os.path.join(base_root, 'right_image')
        self.depth_img_root = os.path.join(base_root, 'depth')
        self.point_cloud_root = os.path.join(base_root, 'point_cloud')
        self.recap_root = os.path.join(base_root, 'recap')

def main():

    base_root = '/media/jetson/Volume/Data'
    data = storing_data(base_root)
    print(data.left_img_root)
    print(data.right_img_root)
    print(data.depth_img_root)
    print(data.point_cloud_root)
    print(data.recap_root)

if __name__ == "__main__":
    main()