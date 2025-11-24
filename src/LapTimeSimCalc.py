"""
---------------------------
Lap Time Simulation Calculator - OLS
---------------------------

This class computes the speed trace given the Performance Envelope and
the track file (track curvature).

---------------------------
@autor: Davide Strassera
@first release: 2019-12-21
by Python 3.7
---------------------------

"""
# Import Packages
import numpy as np
import scipy.interpolate as interp
class LapTimeSimCalc:
    def __init__(self, TrackFile, accEnvDict, prevLap = None):
        # inputs
        self.prevLap = prevLap
        self.track = np.loadtxt(TrackFile)
        # will use previous lap GGVacc and GGVdec if available to save recalculation.
        self.GGVacc = self.prevLap.lapTimeSimDict["GGVacc"] if prevLap else None
        self.GGVdec = self.prevLap.lapTimeSimDict["GGVdec"] if prevLap else None
        self.GGVfull = accEnvDict["GGVfull"]  # ax,ay,vx
        # Gets the starting speed from previous lap if available.
        self.vxaccStart = self.prevLap.lapTimeSimDict["vxaccEnd"] if prevLap else 10
        # outputs
        self.lapTimeSimDict = {
            "vcar": None,
            "dist": None,
            "time": None,
            "laptime": None,
            "vcarmax": None,
            "vxaccEnd": None,
            "vxacc": None,
            "vxdec": None,
            "vxcor": None,
            "GGVacc": None,
            "GGVdec": None,
        }

    @staticmethod
    def splitGGVfull(GGVfull):
        """This method split the GGVfull (which means complete acc, dec and
           mirror left, right), and split it into two matrices: GGVacc
           and GGVdec. This is needed for proper griddata interpolation."""
        GGVacc = np.array([])
        GGVdec = np.array([])
        j, k = 0, 0
        for i in range(len(GGVfull[:, 0])):
            # GGVacc with ax >=0
            if GGVfull[i, 0] >= 0:
                GGVacc = np.concatenate((GGVacc, GGVfull[i, :]))
                j += 1
            # GGVdec with ax < 0 or (with ax=0 and not max speed)
            vxmax = max(GGVfull[:, 2])
            bAxZeroAndNotTopSpeed = GGVfull[i, 0] == 0 and GGVfull[i, 2] < vxmax
            if GGVfull[i, 0] < 0 or bAxZeroAndNotTopSpeed:
                GGVdec = np.concatenate((GGVdec, GGVfull[i, :]))
                k += 1
        # ncol = np.size(GGVfull, axis=1)
        GGVacc = np.resize(GGVacc, (j, 3))
        GGVdec = np.resize(GGVdec, (k, 3))
        return GGVacc, GGVdec

    # GGV surface
    @staticmethod
    def GGVSurfInterpreter(X, Y, Z):
        """ the GGV vectors (X=ax, Y=ay, Z=speed) are used to create an interpolator,
            that in future will receive ay and vx and return ax, based on GGV vectors."""
        ggvInterpreter = interp.LinearNDInterpolator(np.column_stack((Y, Z)), X)  # ,fill_value=0.0)
        return ggvInterpreter     

    def Run(self):
        # Split the full GGV in acc and dec, if not already done
        if not self.prevLap:
            self.GGVacc, self.GGVdec = LapTimeSimCalc.splitGGVfull(self.GGVfull)
        # Load TrackFile
        track = self.track
        dist = track[:, 0]
        curv = track[:, 1]
        # Speed Calculations
        small = 0.00000001  # to avoid division by zero    

        # 1. Max Cornering Speed ---------------------------------------------
        # if previous lap available use its cornering speed
        if not self.prevLap:
            curvvect = np.array([])
            vxvect = np.array([])
            for i in range(len(self.GGVacc[:, 2])):
                # if ax == 0 & ay is positive
                if(self.GGVacc[i, 0] == 0) and (self.GGVacc[i, 1] >= 0):
                    vxclipped = max(self.GGVacc[i, 2], small)  # to avoid div by zero
                    curvvect = np.append(curvvect, self.GGVacc[i, 1]/pow(vxclipped, 2))  # C=ay/v^2
                    vxvect = np.append(vxvect, vxclipped)
            curvvect[0] = 0.5
            curvclipped = np.zeros(len(curv))
            for i in range(len(curv)):
                # curvature clipped to max speed
                curvclipped[i] = max(np.absolute(curv[i]), min(curvvect))
            # v corner from pure lateral (ay)
            vxcor = np.interp(curvclipped, curvvect, vxvect, period=360)
        else:
            vxcor = self.prevLap.lapTimeSimDict["vxcor"]

        # 2. Max Acceleration Speed ------------------------------------------
        vxacc = np.zeros(len(curv))
        vxacc[0] = self.vxaccStart  # must be the last vacc
        ayreal = np.zeros(len(curv))
        axcombine = np.zeros(len(curv))

        X, Y, Z = self.GGVacc[:, 0], self.GGVacc[:, 1], self.GGVacc[:, 2]
        ggvInterp = LapTimeSimCalc.GGVSurfInterpreter( X, Y, Z)
        for i in range(len(dist)-1):
            ayreal[i] = pow(vxacc[i], 2)/(1/max(curv[i], small))
            axcombine[i] = ggvInterp(ayreal[i], vxacc[i])
            vxacc[i+1] = min(vxcor[i+1], (vxacc[i]+(dist[i+1]-dist[i])
                                          / vxacc[i]*axcombine[i]))
            # If a previous lap exists, once it reaches the same speed as the previous lap,
            # it uses previous lap from that point onward to save computation time.
            if self.prevLap:
                lapPointIndex = None
                if vxacc[i+1] == self.prevLap.lapTimeSimDict["vxacc"][i+1]:
                    lapPointIndex = i
                    vxacc[lapPointIndex:] = self.prevLap.lapTimeSimDict["vxacc"][lapPointIndex:]
                    break

        # 3. Max Deceleration Speed ------------------------------------------
        if not self.prevLap:
            vxdec = np.zeros(len(curv))
            vxdec[-1] = vxacc[-1]
            ayreal = np.zeros(len(curv))
            axcombine = np.zeros(len(curv))

            X, Y, Z = self.GGVdec[:, 0], self.GGVdec[:, 1], self.GGVdec[:, 2]
            ggvInterp = LapTimeSimCalc.GGVSurfInterpreter( X, Y, Z)
            for i in reversed(range(len(dist))):
                ayreal[i] = pow(vxdec[i], 2)/(1/max(curv[i], small))
                axcombine[i] = ggvInterp(ayreal[i], vxdec[i])
                vxdec[i-1] = min(vxcor[i-1], (vxdec[i]+(dist[i-1]-dist[i])
                                            / vxdec[i]*axcombine[i]))
            vxdec[-1] = vxacc[-1]
        #if previous lap available use its deceleration speed
        else:
            vxdec = self.prevLap.lapTimeSimDict["vxdec"]

        # Final speed (vcar) ---------------------------------------------
        vcar = np.zeros(len(dist))
        timestep = np.zeros(len(dist))
        time = np.zeros(len(dist))
        for i in range(len(dist)-1):
            vcar[i] = min(vxcor[i], vxacc[i], vxdec[i])
            timestep[i] = (dist[i+1]-dist[i])/vcar[i]
            time[i] = sum(timestep)

        vcar[(len(vcar)-1)] = vcar[(len(vcar)-2)]

        laptime = np.round(max(time), 3)
        vcarmax = np.round(max(vcar), 3)

        self.lapTimeSimDict = {
            "vcar": vcar,
            "dist": dist,
            "time": time,
            "laptime": laptime,
            "vcarmax": vcarmax,
            "vxaccEnd": vcar[(len(vcar) - 1)],
            "vxacc": vxacc,
            "vxdec": vxdec,
            "vxcor": vxcor,
            "GGVacc": self.GGVacc,
            "GGVdec": self.GGVdec,
        }

        print("LapSimTimeCalc completed")
