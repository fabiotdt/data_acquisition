import tkinter as tk
from tkinter import *
from tkinter import ttk
import os

varieties = {
    'ager' : 'agerato',
    'bocl' : 'bocche_leone',
    'bego' : 'begonie',
    'marg' : 'margherite',
    'cale' : 'calendule',
    'borr' : 'borraggine',
    'fior' : 'fiordaliso',
    'dagr' : 'dalie_grandi',
    'dapi' : 'dalie_picole',
    'garo' : 'garofani',
    'fucs' : 'fucsie',
    'impa' : 'impatients',
    'gera' : 'geranei',
    'prim' : 'primule',
    'rose' : 'rose',
    'tage' : 'tagete',
    'vipo' : 'violette_piccole',
    'viga' : 'violette_grandi'
    }

class storing_data():
    def __init__(self, base_root, flowers):
        #self.base_root = '/media/jetson/Volume/Data'
        
        self.left_img = os.path.join(base_root, 'left_image')
        self.right_img = os.path.join(base_root, 'right_image')
        self.depth_img = os.path.join(base_root, 'depth')
        self.point_cloud = os.path.join(base_root, 'point_cloud')
        self.recap_root = os.path.join(base_root, 'recap')
        self.flowers = flowers

    def __len__(self, path):        
        return len(os.listdir(path))
    
    def __split__(self, path):
        return path.split('/')[-1]
    
    def count_flowers(self, path):
        files = [code[:4] for code in os.listdir(path)]
        flowers = {}
        #create a dicionary with the flowers and the number of images
        for key in self.flowers.keys():
            flowers[key] = files.count(key)
        return flowers
    

#base_root = '/media/jetson/Volume/Data'
#base_root = '/home/jetson/Documents/Fabio/Data'
base_root = '/media/jetson/TDTF_sd/Data'
root = storing_data(base_root, varieties)

frame = tk.Tk()
screen_width = frame.winfo_screenwidth()
screen_height = frame.winfo_screenheight()

frame.title("Files summary")
frame.geometry(str(int(screen_width*0.45)) + "x" + str(int(screen_height*0.5)))

def update():
    l_img_num.config(text=str(root.__len__(root.left_img)))
    r_img_num.config(text=str(root.__len__(root.right_img)))
    d_img_num.config(text=str(root.__len__(root.depth_img)))
    p_cloud_num.config(text=str(root.__len__(root.point_cloud)))
    recap_num.config(text=str(root.__len__(root.recap_root)))

    for idk, key in enumerate(varieties):

        number = root.count_flowers(root.left_img)[key]

        key_num = tk.Label(frame, text = str(number), font=("Helvetica"))
        key_num.grid(row=idk, column=5, ipadx=2, ipady=2)

        #key_num.config(text=str(number))  
    
    frame.after(1000, update)


l_img = tk.Label(frame, text="Left image: ", font=("Helvetica"))
l_img.grid(row=0, column=0, ipadx=2, ipady=2)

l_img_num = tk.Label(frame, font=("Helvetica"))
l_img_num.grid(row=0, column=1, ipadx=2, ipady=2)

r_img = tk.Label(frame, text="Right image: ", font=("Helvetica"))
r_img.grid(row=1, column=0, ipadx=2, ipady=2)

r_img_num = tk.Label(frame, font=("Helvetica"))
r_img_num.grid(row=1, column=1, ipadx=2, ipady=2)

d_img = tk.Label(frame, text="Depth image: ", font=("Helvetica"))
d_img.grid(row=2, column=0, ipadx=2, ipady=2)

d_img_num = tk.Label(frame, font=("Helvetica"))
d_img_num.grid(row=2, column=1, ipadx=2, ipady=2)

p_cloud = tk.Label(frame, text="Point cloud: ", font=("Helvetica"))
p_cloud.grid(row=3, column=0, ipadx=2, ipady=2)

p_cloud_num = tk.Label(frame, font=("Helvetica"))
p_cloud_num.grid(row=3, column=1, ipadx=2, ipady=2)

recap = tk.Label(frame, text="Recap: ", font=("Helvetica"))
recap.grid(row=4, column=0, ipadx=2, ipady=2)

recap_num = tk.Label(frame, font=("Helvetica"))
recap_num.grid(row=4, column=1, ipadx=2, ipady=2)

for idk, key in enumerate(varieties):

    key = tk.Label(frame, text=varieties[key]+": " , font=("Helvetica"))
    key.grid(row=idk, column=4, ipadx=2, ipady=2)

    """key_num = tk.Label(frame, font=("Helvetica"))
    key_num.grid(row=idk, column=5, ipadx=2, ipady=2)"""

update()
frame.mainloop()

