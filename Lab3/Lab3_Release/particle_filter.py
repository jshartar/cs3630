#Jordan Shartar and Justin Duan
#

from grid import *
from particle import Particle
from utils import *
from setting import *

import random
import math
import bisect

# ------------------------------------------------------------------------
def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = []
    for p in particles:
        x = p.x
        y = p.y
        h = p.h
        dx = odom[0] + random.gauss(0, ODOM_TRANS_SIGMA)
        dy = odom[1] + random.gauss(0, ODOM_TRANS_SIGMA)
        dh = odom[2] + random.gauss(0, ODOM_HEAD_SIGMA)
        rotdx, rotdy = rotate_point(dx, dy, p.h)
        x = x + rotdx
        y = y + rotdy
        h = h + dh
        newParticle = Particle(x,y,h)
        motion_particles.append(newParticle)
    return motion_particles
# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    if not measured_marker_list:
        return particles
    # Update particle weights
    weights = []
    for p in particles:
        if grid.is_free(*p.xy):
            if measured_marker_list:
                pMarkers = p.read_markers(grid)
                if len(measured_marker_list) == 0 or len(pMarkers) == 0 or len(measured_marker_list) > len(pMarkers):
                    w = 0
                else:
                    # create pairs from markers based on closest distance
                    pairs = []
                    for m in measured_marker_list:
                        closest = 1000000000
                        for pm in pMarkers:
                            distance = grid_distance(m[0],m[1],pm[0],pm[1])
                            if distance < closest:
                                closestParticle = pm
                                closest = distance
                        pairs.append((m, closestParticle))
                        pMarkers.remove(closestParticle)
                        if len(pMarkers) == 0:
                            break
                    #prob calculations that will be w
                    prob = 1
                    for pair in pairs:
                        angleBetweenMarkers = diff_heading_deg(pair[0][2], pair[1][2])
                        distBetweenMarkers = grid_distance(pair[0][0], pair[0][1], pair[1][0], pair[1][1])

                        #
                        #
                        # note to self: check slide 28 of L9_ParticleFilter2.pdf for prob calculations !!!!!!!
                        #
                        #

                        prob *= math.e ** -(
                                distBetweenMarkers ** 2 / (2 * MARKER_TRANS_SIGMA ** 2) + angleBetweenMarkers ** 2 / (
                                    2 * MARKER_ROT_SIGMA ** 2))
                    w = prob
            else:
                w = 1
        else:
            w = 0
        weights.append(w)

    measured_particles = []
    #set really small so it doesnt have anything smaller from weights (not counting weights=0)
    minAvgWeight = 0.0000000001
    # Normalise weights and count number of weights equal to zero
    sumW = 0
    count = 0
    for w in weights:
        sumW += w
        if w == 0:
            count += 1
    avgWeight = sumW / len(particles)

    # new average when not accounting for particles with zero weight
    if count != len(particles):
        avgWeight = avgWeight / (len(particles) - count) * len(particles)

    normWeights = []
    if sumW:
        for w in weights:
            normWeights.append(w / sumW)

    wsum = 0.0
    count = 0
    distribution = []
    for w in normWeights:
        wsum += w
        distribution.append(wsum)
        if w == 0:
            count += 1

    for part in particles:
        if count == len(particles):
            best = None
        else:
            particleCopy = particles
            best = particleCopy[bisect.bisect_left(distribution, random.uniform(0, 1))]
        px = best.x
        py = best.y
        ph = best.h
        if best is None or avgWeight < minAvgWeight:
            particleToAdd = Particle.create_random(1,grid)[0]
        else:
            particleToAdd = Particle(px,py,ph)
        measured_particles.append(particleToAdd)

    return measured_particles




