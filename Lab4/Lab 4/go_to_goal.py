## If you run into an "[NSApplication _setup] unrecognized selector" problem on macOS,
## try uncommenting the following snippet


###############################################################################
#Jordan Shartar & Justin Duan
#
#
#
#we used our implementation of the particle filter
#
#
##############################################################################

try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass

from skimage import color
import cozmo
import numpy as np
from numpy.linalg import inv
import threading
import time
import sys
import asyncio
from PIL import Image

from markers import detect, annotator

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *
from cozmo.util import degrees, radians

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
flag_odom_init = False


# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid, show_camera=True)
pf = ParticleFilter(grid)

def compute_odometry(curr_pose, cvt_inch=True):
    '''
    Compute the odometry given the current pose of the robot (use robot.pose)

    Input:
        - curr_pose: a cozmo.robot.Pose representing the robot's current location
        - cvt_inch: converts the odometry into grid units
    Returns:
        - 3-tuple (dx, dy, dh) representing the odometry
    '''

    global last_pose, flag_odom_init
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees

    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / grid.scale, dy / grid.scale

    return (dx, dy, diff_heading_deg(curr_h, last_h))


async def marker_processing(robot, camera_settings, show_diagnostic_image=False):
    '''
    Obtain the visible markers from the current frame from Cozmo's camera.
    Since this is an async function, it must be called using await, for example:

        markers, camera_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)

    Input:
        - robot: cozmo.robot.Robot object
        - camera_settings: 3x3 matrix representing the camera calibration settings
        - show_diagnostic_image: if True, shows what the marker detector sees after processing
    Returns:
        - a list of detected markers, each being a 3-tuple (rx, ry, rh)
          (as expected by the particle filter's measurement update)
        - a PIL Image of what Cozmo's camera sees with marker annotations
    '''

    global grid

    # Wait for the latest image from Cozmo
    image_event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # Convert the image to grayscale
    image = np.array(image_event.image)
    image = color.rgb2gray(image)

    # Detect the markers
    markers, diag = detect.detect_markers(image, camera_settings, include_diagnostics=True)

    # Measured marker list for the particle filter, scaled by the grid scale
    marker_list = [marker['xyh'] for marker in markers]
    marker_list = [(x/grid.scale, y/grid.scale, h) for x,y,h in marker_list]

    # Annotate the camera image with the markers
    if not show_diagnostic_image:
        annotated_image = image_event.image.resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(annotated_image, markers, scale=2)
    else:
        diag_image = color.gray2rgb(diag['filtered_image'])
        diag_image = Image.fromarray(np.uint8(diag_image * 255)).resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(diag_image, markers, scale=2)
        annotated_image = diag_image

    return marker_list, annotated_image


async def run(robot: cozmo.robot.Robot):

    global flag_odom_init, last_pose
    global grid, gui, pf

    robot.played_goal_animation = False
    robot.found_goal=False
    # start streaming
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    await robot.set_head_angle(cozmo.util.degrees(7)).wait_for_completed()

    # Obtain the camera intrinsics matrix
    fx, fy = robot.camera.config.focal_length.x_y
    cx, cy = robot.camera.config.center.x_y
    camera_settings = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float)

    ###################
    at_goal = False

    while True:

        if robot.is_picked_up:
            robot.stop_all_motors()
            print("picked up")
            pf = ParticleFilter(grid)
            # robot.pf = pf
            robot.found_goal = False
            robot.played_goal_animation = False
            robot.at_goal = False
            if not robot.played_angry_animation:
                await robot.say_text("Stop holding me daddy uhhh daddy please", duration_scalar=0.75,
                                     voice_pitch=1,
                                     num_retries=2).wait_for_completed()
                await robot.set_lift_height(1, 10000).wait_for_completed()
                await robot.set_lift_height(0, 10000).wait_for_completed()
            robot.played_angry_animation = True
            await robot.set_head_angle(degrees(7)).wait_for_completed()
            continue

        currPose = robot.pose
        odom = compute_odometry(currPose)
        markerlist, camera_image = await marker_processing(robot, camera_settings)
        gui.show_camera_image(camera_image)
        mean_estimate = pf.update(odom, markerlist)
        gui.show_camera_image(camera_image)
        last_pose = currPose

        estimatex = mean_estimate[0]
        estimatey = mean_estimate[1]
        estimateh = mean_estimate[2]
        converged = mean_estimate[3]

        # angle = math.radians(mean_estimate[2])
        # forward = (math.cos(angle), math.sin(angle))
        # target_direction = np.subtract(goal[0:2], mean_estimate[0:2])
        # target_angle = math.atan2(target_direction[1], target_direction[0]) - math.atan2(forward[1], forward[0])

        distx = goal[0] - mean_estimate[0]
        disty = goal[1] - mean_estimate[1]
        angle = math.degrees(math.atan2(disty, distx))
        target_angle = diff_heading_deg(angle, mean_estimate[2])

        if robot.played_goal_animation:
            robot.stop_all_motors()
            if robot.is_picked_up:
                robot.stop_all_motors()
                print("picked up")
                pf = ParticleFilter(grid)
                # robot.pf = pf
                robot.found_goal = False
                robot.played_goal_animation = False
                robot.at_goal = False
                if not robot.played_angry_animation:
                    await robot.say_text("daddy stop helping me. senpai ", duration_scalar=0.75,
                                         voice_pitch=1,
                                         num_retries=2).wait_for_completed()
                    await robot.set_lift_height(1, 10000).wait_for_completed()
                    await robot.set_lift_height(0, 10000).wait_for_completed()
                robot.played_angry_animation = True
                await robot.set_head_angle(degrees(7)).wait_for_completed()
                continue
        elif converged:
            #robot.found_goal = True
            await robot.set_head_angle(degrees(7)).wait_for_completed()
            print("converged")
            pos = mean_estimate[0:2]
            rPos = tuple(int(pos[x]) for x in range(len(pos)))
            #robot.found_goal =True
            if robot.found_goal:
                print("foundgoal")
                target_angle = 0

                if robot.is_picked_up:
                    robot.stop_all_motors()
                    print("picked up")
                    pf = ParticleFilter(grid)
                    # robot.pf = pf
                    robot.found_goal = False
                    robot.played_goal_animation = False
                    robot.at_goal = False
                    if not robot.played_angry_animation:
                        await robot.say_text("Stop holding me daddy uhhh daddy please", duration_scalar=0.75,
                                             voice_pitch=1,
                                             num_retries=2).wait_for_completed()
                        await robot.set_lift_height(1, 10000).wait_for_completed()
                        await robot.set_lift_height(0, 10000).wait_for_completed()
                    robot.played_angry_animation = True
                    await robot.set_head_angle(degrees(7)).wait_for_completed()
                    continue
                #dist = math.sqrt((goal[0] - pos[0])**2 + (goal[1] - pos[1])**2)
                # robot.stop_all_motors()
                # robot.drive_straight(distance=dist, speed=10).wait_for_completed()
            else:
                print('foundgoal else')
                # angle = math.radians(mean_estimate[2])
                # forward = (math.cos(angle), math.sin(angle))
                # target_direction = np.subtract(goal[0:2], mean_estimate[0:2])
                # target_angle = math.atan2(target_direction[1], target_direction[0]) - math.atan2(forward[1], forward[0])
                distx = goal[0] - mean_estimate[0]
                disty = goal[1] - mean_estimate[1]
                angle = math.degrees(math.atan2(disty, distx))
                target_angle = diff_heading_deg(angle, mean_estimate[2])

                if robot.is_picked_up:
                    robot.stop_all_motors()
                    print("picked up")
                    pf = ParticleFilter(grid)
                    # robot.pf = pf
                    robot.found_goal = False
                    robot.played_goal_animation = False
                    robot.at_goal = False
                    if not robot.played_angry_animation:
                        await robot.say_text("Stop holding me daddy uhhh daddy please", duration_scalar=0.75,
                                             voice_pitch=1,
                                             num_retries=2).wait_for_completed()
                        await robot.set_lift_height(1, 10000).wait_for_completed()
                        await robot.set_lift_height(0, 10000).wait_for_completed()
                    robot.played_angry_animation = True
                    await robot.set_head_angle(degrees(7)).wait_for_completed()
                    continue

                # robot.stop_all_motors()
                # robot.turn_in_place(degrees(target_angle), num_retries=3).wait_for_completed()
                # robot.found_goal=True
            at_goal = True
            pickupinloop = False

            gPos = tuple(goal[0:2])
            for x in range(len(rPos)):
                if abs(rPos[x] - gPos[x]) > 1:
                    at_goal=False
                    print("forloop at goal")

                    if robot.is_picked_up:
                        robot.stop_all_motors()
                        print("picked up")
                        pf = ParticleFilter(grid)
                        # robot.pf = pf
                        robot.found_goal = False
                        robot.played_goal_animation = False
                        robot.at_goal = False
                        if not robot.played_angry_animation:
                            await robot.say_text("Stop holding me daddy uhhh daddy please", duration_scalar=0.75,
                                                 voice_pitch=1,
                                                 num_retries=2).wait_for_completed()
                            await robot.set_lift_height(1, 10000).wait_for_completed()
                            await robot.set_lift_height(0, 10000).wait_for_completed()
                        robot.played_angry_animation = True
                        await robot.set_head_angle(degrees(7)).wait_for_completed()
                        pickupinloop = True
                        break
                    #break
            if pickupinloop:
                continue

            if at_goal:
                print("at goal")
                robot.stop_all_motors()
                await robot.turn_in_place(degrees(-mean_estimate[2]), num_retries=3).wait_for_completed()
                robot.found_goal = True
                #if not robot.played_goal_animation:
                robot.say_text("yeet yeet yeet I DID IT DADDY", play_excited_animation=True, duration_scalar=0.75, voice_pitch=1)
                robot.played_goal_animation = True
                #await robot.play_anim_trigger(cozmo.anim.Triggers.MajorWin)
            elif abs(target_angle) > 2 and abs(2*math.pi - abs(target_angle)) > 2:
                print("elif abs(target)angle)")
                robot.stop_all_motors()
                await robot.turn_in_place(degrees(target_angle), num_retries=2).wait_for_completed()
                robot.found_goal =True
                robot.stop_all_motors()
                await robot.set_head_angle(degrees(7)).wait_for_completed()
                if robot.is_picked_up:
                    robot.stop_all_motors()
                    print("picked up")
                    pf = ParticleFilter(grid)
                    # robot.pf = pf
                    robot.found_goal = False
                    robot.played_goal_animation = False
                    robot.at_goal = False
                    if not robot.played_angry_animation:
                        await robot.say_text("Stop holding me daddy uhhh daddy please", duration_scalar=0.75,
                                             voice_pitch=1,
                                             num_retries=2).wait_for_completed()
                        await robot.set_lift_height(1, 10000).wait_for_completed()
                        await robot.set_lift_height(0, 10000).wait_for_completed()
                    robot.played_angry_animation = True
                    await robot.set_head_angle(degrees(7)).wait_for_completed()
                    continue
            else:
                print("else of at goal")
                robot.found_goal = False
                dist = math.sqrt((goal[0] - pos[0]) ** 2 + (goal[1] - pos[1]) ** 2)
                # if not robot.found_goal:
                #     robot.stop_all_motors()
                #     await robot.turn_in_place(angle=degrees(target_angle),num_retries=2).wait_for_completed()
                #robot.found_goal = True
                robot.stop_all_motors()
                await robot.drive_straight(distance=cozmo.util.distance_mm(dist * 25), speed=cozmo.util.speed_mmps(75)).wait_for_completed()
                #await robot.drive_wheels(20, 20, duration=6)

                if robot.is_picked_up:
                    robot.stop_all_motors()
                    print("picked up")
                    pf = ParticleFilter(grid)
                    # robot.pf = pf
                    robot.found_goal = False
                    robot.played_goal_animation = False
                    robot.at_goal = False
                    if not robot.played_angry_animation:
                        await robot.say_text("daddy stop helping me. senpai ", duration_scalar=0.75,
                                             voice_pitch=1,
                                             num_retries=2).wait_for_completed()
                        await robot.set_lift_height(1, 10000).wait_for_completed()
                        await robot.set_lift_height(0, 10000).wait_for_completed()
                    robot.played_angry_animation = True
                    await robot.set_head_angle(degrees(7)).wait_for_completed()
                    continue
        else:

            await robot.set_head_angle(degrees(7)).wait_for_completed()
            await robot.drive_wheels(-10, 10, duration=0)
            if robot.is_picked_up:
                robot.stop_all_motors()
                print("picked up")
                pf = ParticleFilter(grid)
                # robot.pf = pf
                robot.found_goal = False
                robot.played_goal_animation = False
                robot.at_goal = False
                if not robot.played_angry_animation:
                    await robot.say_text("Stop holding me daddy uhhh daddy please", duration_scalar=0.75, voice_pitch=1,
                                         num_retries=2).wait_for_completed()
                    await robot.set_lift_height(1, 10000).wait_for_completed()
                    await robot.set_lift_height(0, 10000).wait_for_completed()
                robot.played_angry_animation = True
                await robot.set_head_angle(degrees(7)).wait_for_completed()
                continue
            print("turning")

        if robot.is_picked_up:
            robot.stop_all_motors()
            print("picked up")
            pf = ParticleFilter(grid)
            #robot.pf = pf
            robot.found_goal = False
            robot.played_goal_animation = False
            robot.at_goal = False
            if not robot.played_angry_animation:
                await robot.say_text("Stop holding me daddy uhhh daddy please", duration_scalar=0.75, voice_pitch=1,
                                     num_retries=2).wait_for_completed()
                await robot.set_lift_height(1, 10000).wait_for_completed()
                await robot.set_lift_height(0, 10000).wait_for_completed()
            robot.played_angry_animation = True
            await robot.set_head_angle(degrees(7)).wait_for_completed()
            continue
        else:
            robot.played_angry_animation = False

        gui.show_particles(pf.particles)
        gui.show_mean(*mean_estimate)
        gui.updated.set()

    ###################

class CozmoThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)



if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    gui.show_particles(pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()
