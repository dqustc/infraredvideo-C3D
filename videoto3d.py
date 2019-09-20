import numpy as np
import cv2


class Videoto3D:

    def __init__(self, width, height, depth, dataset):
        self.width = width
        self.height = height
        self.depth = depth
        self.dataset = dataset

    def video3d(self, filename, color=False, skip=True):
        cap = cv2.VideoCapture(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if skip:
            frames = [x * nframe / self.depth for x in range(self.depth)]
        else:
            if nframe > self.depth * 4:
                frame_start = nframe / 2 - self.depth * 2
                frame_start = np.floor(frame_start)
                frames = [frame_start + 4 * x for x in range(self.depth)]
            else :
                frame_start = nframe / 2 - self.depth / 2
                frame_start = np.floor(frame_start)
                frames = [frame_start + x for x in range(self.depth)]
        framearray = []

        faction = 3
        Nclip = np.floor(nframe / (faction*self.depth)).astype(np.int)
        nclip = min(Nclip, 2)
        start_clip = np.floor(Nclip / 2 - nclip / 2)
        start_frame = start_clip * faction
        for j in range(nclip):
            for i in range(self.depth):
                cap.set(cv2.CAP_PROP_POS_FRAMES,  start_frame + faction*(j*self.depth + i))
                ret, frame = cap.read()
                if ret == False:
                    cap.release()
                    if len(framearray) >= self.depth:
                        return np.array(framearray)
                    else:
                        return []
                frame = cv2.resize(frame, (self.height, self.width))
                if color:
                    framearray.append(frame)
                else:
                    framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        cap.release()
        return np.array(framearray)



    def get_UCF_classname(self, filename):
        return filename[filename.find('_') + 1:filename.find('_', 2)]

    def get_HMDB_classname(self, dirname):
        return dirname

