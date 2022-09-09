import os
import shutil

import boto3
import cv2
import matplotlib.pyplot as plt

import credentials


output_dir = './output_lane_crossing'
anns_dir = os.path.join(output_dir, 'anns')
imgs_dir = os.path.join(output_dir, 'imgs')

for dir_ in [output_dir, anns_dir, imgs_dir]:
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.mkdir(dir_)

reko_client = boto3.client('rekognition',
                           aws_access_key_id=credentials.access_key,
                           aws_secret_access_key=credentials.secret_key)

input_file = '/media/phillip/FELIPE/youtube/computer_vision/08_illegal_lane_crossing_detection/code/data/Coche - 2165.mp4'
cap = cv2.VideoCapture(input_file)

counter = 0

ret = True
class_names = []
while ret:
    cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
    ret, frame = cap.read()
    if ret:

        H, W, _ = frame.shape

        anns_file = open(os.path.join(anns_dir, '{}.txt'.format(str(counter).zfill(6))), 'w')

        tmp_filename = './tmp.jpg'
        cv2.imwrite(tmp_filename, frame)
        cv2.imwrite(os.path.join(imgs_dir, '{}.jpg'.format(str(counter).zfill(6))), frame)

        with open(tmp_filename, 'rb') as image:
            response = reko_client.detect_labels(Image={'Bytes': image.read()})

        for label in response['Labels']:
            if len(label['Instances']) > 0:
                name = label['Name']
                if name not in class_names:
                    class_names.append(name)
                for instance in label['Instances']:
                    conf = float(instance['Confidence']) / 100
                    w = instance['BoundingBox']['Width']
                    h = instance['BoundingBox']['Height']
                    x = instance['BoundingBox']['Left']
                    y = instance['BoundingBox']['Top']
                    # print(x, y, w, h, conf)
                    # class_index, xc, yc, w, h, confidence
                    anns_file.write('{} {} {} {} {} {}\n'.format(class_names.index(name),
                                                                 x + (w / 2),
                                                                 y + (h / 2),
                                                                 w,
                                                                 h,
                                                                 conf))
                    """
                    x_ = int(x * W)
                    w_ = int(w * W)
                    y_ = int(y * H)
                    h_ = int(h * H)
                    frame = cv2.rectangle(frame, (x_, y_), (x_ + w_, y_ + h_), (0, 255, 0), 2)
                    """

        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # plt.show()

        os.remove(tmp_filename)

        anns_file.close()

        counter += 1

with open(os.path.join(output_dir, 'class.names'), 'w') as fw:
    for name in class_names:
        fw.write('{}\n'.format(name))
    fw.close()
