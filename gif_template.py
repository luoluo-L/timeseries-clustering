import imageio
import os

image_folder = 'num_clusters_figs'

#image_folder = os.fsencode(path)

filenames = []

for file in os.listdir(image_folder):
    filename = os.path.join(image_folder,file)
    if filename.lower().endswith( ('.jpg', '.png', '.gif') ):
        filenames.append(filename)

filenames_list = filenames.copy()
    #[filenames[1] , filenames[1],  filenames[0]]

#filenames.sort() # this iteration technique has no built in order, so sort the frames

images = list(map(lambda filename: imageio.imread(filename), filenames_list))

imageio.mimsave(os.path.join('num_clusteres.gif'), images, duration = 1) # modify duration as needed