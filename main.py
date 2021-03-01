import cv2
import numpy as np
import math

face_onnx_path = r"weights/face-RFB-320_simplified.onnx"
smoke_onnx_path = r'weights/smoke.onnx'
label_path = 'labels.txt'


def clip(x, y):
    # [0, 1]
    return max(0, min(x, y))


def nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # score从大到小的索引值
    # order = np.argsort(-scores)  # 也可以

    keep = []
    while order.size > 0:
        i = order[0]  # 得到第一个最大的索引值
        keep.append(i)  # 保留得分最大的索引值
        # 得到中间inter矩形的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])  # x1[i]和除了最大的值之外的值作比较
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # 第i个box和其它box的iou

        # 大于阈值的就不管了（去除掉），小于阈值的就可能是另一个目标框，留下来继续比较
        inds = np.where(ovr <= thresh)[0]  # 返回满足条件的order[1:]中的索引值
        order = order[inds + 1]  # +1得到order中的索引值

    return keep

class FaceDetector:
    def __init__(self, model_path):
        self.strides = [8.0, 16.0, 32.0, 64.0]
        self.min_boxes = [
            [10.0, 16.0, 24.0],
            [32.0, 48.0],
            [64.0, 96.0],
            [128.0, 192.0, 256.0]]

        self.in_h, self.in_w = (240, 320)
        self.face_detector = cv2.dnn.readNetFromONNX(model_path)

        # generate_prior_anchor
        w_h_list = [self.in_w, self.in_h]
        featuremap_size = []
        for size in w_h_list:
            fm_item = []
            for stride in self.strides:
                fm_item.append(np.ceil(size / stride))
            featuremap_size.append(fm_item)

        shrinkage_size = []
        for size in w_h_list:
            shrinkage_size.append(self.strides)

        self.priors = []
        for index in range(4):
            scale_w = self.in_w / shrinkage_size[0][index]
            scale_h = self.in_h / shrinkage_size[1][index]
            for j in range(int(featuremap_size[1][index])):
                for i in range(int(featuremap_size[0][index])):
                    x_center = (i + 0.5) / scale_w
                    y_center = (j + 0.5) / scale_h
                    for k in self.min_boxes[index]:
                        w = k / self.in_w
                        h = k / self.in_h
                        self.priors.append([clip(x_center, 1), clip(y_center, 1),
                                            clip(w, 1), clip(h, 1)])


    def postprocess(self, image_w, image_h, scores, boxes, score_threshold):
        bbox_value = boxes.flatten()
        score_value = scores.flatten()
        num_anchors = len(self.priors)
        # print(bbox_value.shape)
        # print(score_value.shape)

        rect_boxes = []
        confidences = []
        for i in range(num_anchors):
            score = score_value[2 * i + 1]
            if score > score_threshold:
                x_center = bbox_value[i * 4] * 0.1 * self.priors[i][2] + self.priors[i][0]
                y_center = bbox_value[i * 4 + 1] * 0.1 * self.priors[i][3] + self.priors[i][1]
                w = math.exp(bbox_value[i * 4 + 2] * 0.2) * self.priors[i][2]
                h = math.exp(bbox_value[i * 4 + 3] * 0.2) * self.priors[i][3]

                x1 = int(clip(x_center - w / 2.0, 1) * image_w)
                y1 = int(clip(y_center - h / 2.0, 1) * image_h)
                x2 = int(clip(x_center + w / 2.0, 1) * image_w)
                y2 = int(clip(y_center + h / 2.0, 1) * image_h)
                score = clip(score, 1)
                rect_boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(score))

        indices = cv2.dnn.NMSBoxes(rect_boxes, confidences, score_threshold, 0.5)

        if len(indices):
            indices = indices.flatten()
        rect_boxes = np.array(rect_boxes)[indices]
        confidences = np.array(confidences)[indices]
        # keep = self.nms(rect_boxes.astype(np.int32), confidences, 0.5)
        # print(rect_boxes[indices])
        # print(confidences[indices])
        return rect_boxes, confidences


    def __call__(self, img, **kwargs):

        inputBlob = cv2.dnn.blobFromImage(img, 1.0 / 128, (320, 240), (127, 127, 127), swapRB=True)
        self.face_detector.setInput(inputBlob)

        scores, boxes = self.face_detector.forward(["scores", "boxes"])
        # print(scores)
        image_h, image_w = img.shape[:2]
        rect_boxes, confidences = self.postprocess(image_w, image_h, scores, boxes, 0.6)

        return rect_boxes, confidences


class SmokeDetector:
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        self.classes = ['smoke']
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        # num_classes = len(self.classes)
        num_classes = 1
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = [np.zeros(1)] * self.nl  # init grid
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)

        self.net = cv2.dnn.readNet(model_path)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

    def postprocess(self, image_w, image_h, outs):

        ratioh, ratiow = image_h / 640, image_w / 640
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold and detection[4] > self.objThreshold:
                    center_x = int(detection[0] * ratiow)
                    center_y = int(detection[1] * ratioh)
                    width = int(detection[2] * ratiow)
                    height = int(detection[3] * ratioh)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        # print(indices)
        if len(indices):
            indices = indices.flatten()

        boxes = np.array(boxes)[indices]
        confidences = np.array(confidences)[indices]

        return boxes, confidences


    def __call__(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (640, 640), [0, 0, 0], swapRB=True, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = outs[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # outs[i] = outs[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            outs[i] = outs[i].reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            if self.grid[i].shape[2:4] != outs[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = 1 / (1 + np.exp(-outs[i]))  ### sigmoid
            # 其实只需要对x,y,w,h做sigmoid变换的， 不过全做sigmoid变换对结果影响不大，
            # 因为sigmoid是单调递增函数，那么就不影响类别置信度的排序关系，因此不影响后面的NMS
            # 不过设断点查看类别置信度，都是负数，看来有必要做sigmoid变换把概率值强行拉回到0到1的区间内
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * int(self.stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, self.no))
        z = np.concatenate(z, axis=1)

        image_h, image_w = srcimg.shape[:2]
        rect_boxes, confidences = self.postprocess(image_w, image_h, z)

        return rect_boxes, confidences


class SmokerDetector:
    def __init__(self, face_model, smoke_model, label_path, face_margin=5):
        self.face_detector = FaceDetector(face_model)
        self.smoke_detector = SmokeDetector(smoke_model)
        with open(label_path, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.margin = face_margin


    def drawPred(self, frame, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % ('smoke', label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])),
        #                     (left + round(1.5 * labelSize[0]), top + baseLine),
        #                     (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), thickness=2)

        return frame

    def __call__(self, image_path):
        orig_image = cv2.imread(image_path)
        dst_image = orig_image.copy()

        face_boxes, _ = self.face_detector(orig_image)
        #
        image_h, image_w = orig_image.shape[:2]
        # print(rect_boxes)
        face_imgs = []
        for box in face_boxes:
            x1, y1, w, h = box

            xx1 = max(int(x1 - self.margin * w / 10), 0)
            yy1 = max(int(y1 - self.margin * h / 10), 0)
            xx2 = min(int(x1 + w + self.margin * w / 10), image_w)
            yy2 = min(int(y1 + h + self.margin * h / 10), image_h)

            face_img = orig_image[yy1:yy2, xx1:xx2]

            smoke_boxes, smoke_confidences = self.smoke_detector(face_img)

            if len(smoke_boxes) and len(smoke_confidences):
                smoke_boxes[:,0] += xx1
                smoke_boxes[:,1] += yy1

            for i, box in enumerate(smoke_boxes):
                dst_image = self.drawPred(dst_image, smoke_confidences[i],
                                          box[0], box[1], box[0]+box[2], box[1]+box[3])

        return dst_image


if __name__ == "__main__":

    smoker_detector = SmokerDetector(face_onnx_path, smoke_onnx_path, label_path)
    frame = smoker_detector('images/smoke26.jpg')

    winName = 'Smoke Detection'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




