import cozmo
import numpy as np
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
from cozmo.util import distance_mm, speed_mmps
import sys

def run(sdk_conn):
    #set up cozmo robot and images/training
    robot = sdk_conn.wait_for_robot()
    imClass = ImageClassifier()
    images = imClass.getImages()

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    state = "none"

    #finite state machine
    while(True):
        if (state == "none"):
            print("completed task going to idle")
            state = idle(robot, images)
            print("current: idle")
        elif(state == "drone"):
            drone(robot)
            state = "none"
        elif(state == "order"):
            order(robot)
            state = "none"
        elif(state == "inspection"):
            inspection(robot)
            state = "none"
        else:
            print("invalid state: ", state)
            print("returning to idle")
            state = "none"

def idle(robot, images):
    #set robot to default
    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    robot.set_lift_height(height=0, accel=0, max_speed=0.5).wait_for_completed()
    #images from cozmo
    cozmoImages = []
    #labeled images
    labels = {}
    #test array of images for prediction
    test = []
    #boolean for while loop
    go = True
    #get images to determine if one of the commands is found
    while go:
        image = robot.world.latest_image
        if image:
            cozmoImages.append(robot.world.latest_image)
        robot.turn_in_place(angle=cozmo.util.Angle(degrees=10), speed=cozmo.util.Angle(degrees=10), in_parallel=True).wait_for_completed()
        image = robot.world.latest_image
        if image:
            cozmoImages.append(robot.world.latest_image)
        robot.turn_in_place(angle=cozmo.util.Angle(degrees=-10), speed=cozmo.util.Angle(degrees=10), in_parallel=True).wait_for_completed()
        if len(cozmoImages) >= 4:
            go = False
    #adds images to test array for feature extraction and predicting
    for im in cozmoImages:
        test.append(np.array(im.raw_image))
    features = images.extract_image_features(test)
    test = images.predict_labels(features)
    #gets amount of each label found
    for t in test:
        if labels.get(t) is None:
            labels[t] = 1
        else:
            labels[t] += 1
    key = list(labels.keys())
    value = list(labels.values())
    prediction = key[value.index(max(value))]
    #only if labels are conclusive do action else rotate and go back to idle
    if labels[prediction] >= 3:
        return prediction
    else:
        idle(robot, images)
    #else:
        #rotate if nothing was found to look around arena for commands
        #robot.turn_in_place(angle=cozmo.util.Angle(degrees=10), speed=cozmo.util.Angle(degrees=10), in_parallel=True).wait_for_completed()
        #idle(robot, images)

def drone(robot):
    #say drone
    print("current: drone")
    robot.say_text("drone").wait_for_completed()
    #find cube
    cube = robot.world.wait_for_observed_light_cube(include_existing=True)
    #pick up cube and go forward 10cm
    robot.pickup_object(cube, num_retries=10).wait_for_completed()
    robot.drive_straight(distance_mm(100), speed_mmps(50)).wait_for_completed()
    #drop cube
    robot.place_object_on_ground_here(cube).wait_for_completed()
    #dive back 10cm
    robot.drive_straight(distance_mm(-100),speed_mmps(50)).wait_for_completed()
    robot.say_text("completed action").wait_for_completed()
def order(robot):
    # say order
    print("current: order")
    robot.say_text("order").wait_for_completed()
    #drive in circle radius 10cm then return to idle
    robot.drive_wheels(l_wheel_speed=45, r_wheel_speed=15, duration=20)
    robot.say_text("completed action").wait_for_completed()
def inspection(robot):
    # say inspection
    print("current: inspection")
    robot.say_text("inspection").wait_for_completed()
    #go = True
    #while(go):

    #    robot.set_lift_height(height=1, accel=0, max_speed=0.4, in_parallel=True)

    for x in range(4):
        #robot.set_lift_height(height=0, accel=0, max_speed=0.4, in_parallel=True)

        robot.set_lift_height(height=1, accel=0, max_speed=0.4, in_parallel=True)
        robot.drive_straight(distance_mm(100), speed_mmps(50), in_parallel=True).wait_for_completed()
        robot.set_lift_height(height=0, accel=0, max_speed=0.4, in_parallel=True)
        robot.set_lift_height(height=1, accel=0, max_speed=0.4, in_parallel=True)#.wait_for_completed()
        robot.drive_straight(distance_mm(100), speed_mmps(50), in_parallel=True).wait_for_completed()
        robot.set_lift_height(height=0, accel=0, max_speed=0.4, in_parallel=True)
        robot.turn_in_place(angle=cozmo.util.Angle(degrees=90), speed=cozmo.util.Angle(degrees=90), in_parallel=True).wait_for_completed()
    robot.set_lift_height(height=0, accel=0, max_speed=0.5, in_parallel=True).wait_for_completed()
    robot.say_text("completed action").wait_for_completed()

#image classifier from lab 1
class ImageClassifier:

    def __init__(self):
        self.classifer = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir + "*.bmp", load_func=self.imread_convert)

        # create one large array of image data
        data = io.concatenate_images(ic)

        # extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return (data, labels)

    def extract_image_features(self, data):
        l = []
        for im in data:
            im_gray = color.rgb2gray(im)

            im_gray = filters.gaussian(im_gray, sigma=0.4)

            f = feature.hog(im_gray, orientations=10, pixels_per_cell=(48, 48), cells_per_block=(4, 4),
                            feature_vector=True, block_norm='L2-Hys')
            l.append(f)

        feature_data = np.array(l)
        return feature_data

    def train_classifier(self, train_data, train_labels):
        self.classifer = svm.LinearSVC()
        self.classifer.fit(train_data, train_labels)

    def predict_labels(self, data):
        predicted_labels = self.classifer.predict(data)
        return predicted_labels

    def test(self, train_data, train_labels):
        self.train_classifier(train_data, train_labels)
        predicted_labels = self.predict_labels(train_data)
        print("\nTraining results")
        print("=============================")
        print("Confusion Matrix:\n", metrics.confusion_matrix(train_labels, predicted_labels))
        print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
        print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))

    def getImages(self):
        (train_raw, train_labels) = self.load_data_from_folder("./train/")
        train_data = self.extract_image_features(train_raw)
        self.train_classifier(train_data, train_labels)
        self.predict_labels(train_data)
        return self

if __name__ == '__main__':
    cozmo.setup_basic_logging()

    try:
        cozmo.connect(run)
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)