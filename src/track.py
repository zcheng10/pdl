#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import cv2
from ultralytics import YOLO

from src.pnn import *

nround = lambda x : np.round(x, 2)

CLASS_BALL = 32

class Watcher:
    """Given a series of images, predict the trajectory
    of the next several frames
    """

    def __init__(self, modelPath = "../model/yolo11s.pt") -> None:
        """Load the model
        """
        self.model = YOLO(modelPath)

    def __call__(self, *args, **kwds):
        pass

    def watch(self, imgs, showWindow = None) -> list[torch.tensor]:
        """Find the bounding boxes in these images

        Returns:
            The list of boxes, for each image, there may be more than
            1 boxes
        """
        lbox = []
        for f in imgs:
            results = self.model.predict(f, 
                    classes = [CLASS_BALL], verbose = False)
            for r in results:
                bb = r.boxes.xyxy
                # r.boxes.xywh: xy is center
                lbox.append(bb)
                # if bb.shape[0] == 0:
                #     print("No", end = ".. ")
                # print("box =", bb)
                # print("obb =", r.obb, ", probs =", r.probs)

            if showWindow is not None:
                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Display the annotated frame
                cv2.imshow(showWindow, annotated_frame)

        return lbox
    
    @staticmethod
    def hasBBox(bb : torch.tensor) -> bool:
        """Whether this tensor corresponds to
        a bounding box
        """
        return bb.shape[0] >= 1 


class Feeder:
    """Feed video to the water and get the predictions        
    """
    def __init__(self, video_path, max_frames = None, 
                 ref_path = None, verbose = False) -> None:
        """Extract bboxes of sports balls from this viedoe
        """
        # video_path = "test/ext.webm"
        cap = cv2.VideoCapture(video_path)
        self.wt = Watcher()
        self.video = video_path
        
        # Loop through the video frames
        if max_frames is None:
            max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.marked = dict()
        self.predicted = dict()

        if ref_path is not None:
            # -- read marked from the file
            if ref_path == "":
                ref_path = Feeder.defaultRefName(video_path)
            self.marked = Feeder.loadBoxes(ref_path)
            self.filterBoxes()
            return

        cnt = 0
        while cap.isOpened() and cnt < max_frames:
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                lbox = self.wt.watch([frame])
                if Watcher.hasBBox(lbox[0]):
                    self.marked[cnt]= lbox[0]
                
                if verbose:
                    print(cnt, "->", lbox)

            if cnt % 100 == 0:
                print("read frame", cnt)
            cnt += 1
        cap.release()

        # -- write marked
        ref_file = Feeder.defaultRefName(video_path)
        Feeder.saveBoxed(self.marked, ref_file)

    def predict(self, seq: int = 8, verbose: bool = False):
        """Use a sequence to prdict the next one
        """
        mk = sorted(list(self.marked.keys()))
        obs = [[c, self.marked[c]] for c in mk]
        m = MotionSolver()
        pose = None
        predictions = 1

        # -- start prediction
        num, i = len(obs), seq
        while i <= num:
            # input [i - seq, i), a tensor [seq, 4]
            input = [obs[j][1].flatten().tolist() for j in range(i-seq, i)]
            input = torch.tensor(input, requires_grad=False)

            pose, next_states, next_poses = m.solve(input, guess=pose, 
                        predictions=predictions)
            n_states = next_states.reshape([predictions, 4])

            if verbose:
                print(i, "=> Estimated state =", 
                    nround(pose.detach().numpy()))
                print("     Next states: ", 
                    nround(next_states.detach().numpy()))
            
            
            fn = obs[i-1][0] + 1
            self.predicted[fn] = n_states
            print("predict ", fn, n_states)

            if (i < num) and (fn < obs[i][0]):
                obs.insert(i, [fn, n_states])
                num = len(obs)
           
            i += 1


    def playWithAnnotation(self, winTitle = None, outfile = None):
        """Show the video with annotated bbox

        Args:
            winTitle: the window to show images. If None, images will
                not show

            outfile: if not None, output the annotated frames to
                the video file. If empty (""), use the default
                video file name ("*_boxed.mp4")
        """
        cap = cv2.VideoCapture(self.video)
        if not cap.isOpened():
            return
        
        if winTitle is None:
            winTitle = "Soccer Tracking"

        toWrite = (outfile is not None)
        if toWrite:
            if outfile == "":
                outfile = Feeder.defaultBoxName(self.video)
                
            # Get video properties
            fps = 8  # Frames per second
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

            # Create a VideoWriter object for the output video
            out = cv2.VideoWriter(outfile,
                    fourcc, fps, (width, height))

        cnt = 0
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                if cnt in self.marked:
                    bbox = self.marked[cnt].detach().numpy()
                    bbox = bbox.astype("int")
                    Feeder.annotate(frame, bbox)

                if cnt in self.predicted:
                    bbox = self.predicted[cnt].detach().numpy()
                    bbox = bbox.astype("int")
                    Feeder.annotate(frame, bbox, color = (0, 255, 0))
                
                if toWrite:
                    out.write(frame)
                else:
                    cv2.imshow(winTitle, frame)
                    # Break the loop if 'q' is pressed
                    cv2.waitKey(0)
            else:
                break

            cnt += 1

        cv2.destroyAllWindows()
        cap.release()

        if toWrite:
            out.release()

    def compact(self, video_out_prefix : str, max_sep = 30):
        """Compact the video so that only the ball scenes are kept
        """
        # -- break the marked into clips, and for each clip,
        # add missing frames
        mk = list(self.marked.keys())
        mk = sorted(mk)
        num = len(mk)

        if num == 0:
            return
        
        c1, c2 = mk[0], -1
        toWrite = []
        for i in range(num - 1):
            if mk[i] + max_sep < mk[i + 1]:
                # break here
                c2 = mk[i]
                toWrite.append([c1, c2])
                c1 = mk[i+1]
                c2 = -1

        c2 = mk[num - 1]
        toWrite.append([c1, c2])

        # write clips        
        cap = cv2.VideoCapture(self.video)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

        # Create a VideoWriter object for the output video
        for k in range(len(toWrite)):
            fname = video_out_prefix + "_" + str(k) +".mp4"
            print("writing ", fname)
            out = cv2.VideoWriter(fname,
                    fourcc, fps, (width, height))

            # Define frame range
            start_frame, end_frame = toWrite[k]
            
            # Read and process frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            while cap.isOpened() and start_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check if the current frame is within the desired range
                if start_frame <= end_frame:
                    out.write(frame)  # Write the frame to the output video

                start_frame += 1

            # Release resources
            out.release()
            
        cap.release()
       
    def filterBoxes(self) -> None:
        """Sometimes a frame has 2 or more boxes, need to keep only 1. This is done
          by comparing with the neighboring frames
        """
        mk = sorted(list(self.marked.keys()))
        seq = [ [c, self.marked[c]] for c in mk]
        num = len(seq)
        for i in range(num):
            if seq[i][1].shape[0] == 1:
                continue
            # -- check with i - 1
            if i > 0 and seq[i-1][1].shape[0] == 1:
                k = Feeder.minDistance(seq[i][1], seq[i-1][1])
                seq[i][1] = seq[i][1][k, :].unsqueeze(0)
                continue

            # -- check with i + 1
            if i < num - 1 and seq[i+1][1].shape[0] == 1:
                k = Feeder.minDistance(seq[i][1], seq[i+1][1])
                seq[i][1] = seq[i][1][k, :].unsqueeze(0)
                # print("filter", seq[i][0], "->", seq[i][1])
            
        self.marked.clear()
        cnt = 0
        for s in seq:
            if s[1].shape[0] > 1:
                cnt += 1
            else:
                self.marked[s[0]] = s[1]
            
        print("after filtering, still", cnt, "frames contain >= 2 boxes")
        # print(self.marked)


    @staticmethod
    def minDistance(t1: torch.tensor, t2:torch.tensor) -> int:
        """t1 shape [k, 4], t2 shape [1, 4], find which
        0<= p < k, t1[p, :] is closest to t2 
        """
        a = t1 - t2
        b = torch.sum(a**2, axis = 1)
        return torch.topk(-b, 1).indices.item()

    @staticmethod
    def annotate(img, boxes, color = (0, 0, 255)):
        """Add bboxes to the image
        """
        for i in range(boxes.shape[0]):
            # w, h = int(boxes[i][2]/2), int(boxes[i][3]/2)
            # pt1 = (boxes[i][0] - w, boxes[i][1] - h)
            # pt2 = (boxes[i][0] + w, boxes[i][1] + h )
            pt1 = (boxes[i][0], boxes[i][1])
            pt2 = (boxes[i][2], boxes[i][3])
            cv2.rectangle(img, pt1, pt2, color = color)


    @staticmethod
    def loadBoxes(file):
        """Load <cnt, bboxes> from the given file
        """
        fp = open(file)
        text = fp.read()
        fp.close()
        lst = text.split("\n")
        marked, i = dict(), 0
        while i < len(lst):
            a = lst[i].strip().split(" ")
            if len(a) != 2:
                i += 1; continue
            
            cnt, num = [int(x) for x in a]
            data = [0] * num
            for j in range(i + 1, i + num + 1):
                a = lst[j].split(" ")
                data[j - i - 1] = [float(x) for x in a]
            b = torch.tensor(data, dtype = float)
            marked[cnt] = b
            i += num + 1
        return marked

    @staticmethod
    def saveBoxed(marked, file):
        fp = open(file, "w")
        mk = sorted(list(marked.keys()))
        print("marked =", marked)
        print("mk =", mk)

        for cnt in mk:
            t = marked[cnt]
            fp.write(str(cnt) + " " + str(t.shape[0]) + "\n")
            for i in range(t.shape[0]):
                a = t[i, :].tolist()
                fp.write(" ".join([str(x) for x in a]) + "\n")
        fp.close()

    @staticmethod
    def defaultRefName(video_path):
        return video_path.replace(".mp4", "") + "_ref.txt"
    
    @staticmethod
    def defaultBoxName(video_path):
        return video_path.replace(".mp4", "") + "_boxed.mp4"
