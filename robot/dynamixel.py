#!/usr/bin/env python
# (c) 2022 Wouter Caarls, PUC-RIO

"""Access Dynamixel servos (Protocol 1.0) through Arbotix-M board."""

from math import pi, copysign
from serial import Serial

def checksum(data):
    """Calculate Dynamixel checksum over data."""
    return 255 - sum(data) % 256


def makelist(ids, n=1):
    """Ensure ids is a list of length n."""
    if not hasattr(ids, "__len__"):
        return [ids] * n
    elif len(ids) == 1:
        return ids * n
    else:
        return ids


def pos2rad(pos):
    """Convert Dynamixel servo position to radians."""
    return (pos-512)/1023 * (5/3 * pi)


def rad2pos(angle):
    """Convert radians to Dynamixel servo position."""
    return min(max(int(angle / (5/3*pi) * 1023 + 512), 0), 1023)


def rads2speed(rads):
    """Convert radians per second to Dynamixel servo speed."""
    return min(max(int(rads / (0.111 / 60 * 2 * pi)), 1), 1023)


class Arbotix():
    """Interface to Arbotix-M board."""

    def __init__(self, port=None):
        self.port = None
        if port is not None:
            self.open(port)

    def open(self, port):
        """Open serial port.

        Parameters
        ----------
        port : string
            Name of port to open.
        """
        self.port = Serial(port, 115200, timeout=1)

    def close(self):
        """Close serial port."""
        self.port.close()
        self.port = None

    def read(self, ids, addr, n):
        """Read multiple Dynamixel control table values.

        Parameters
        ----------
        ids : int or list of int
            Dynamixel IDs to read from.
        addr : int
            Start of control table entries to read.
        n : int
            Number of control table entries to read.

        Returns
        -------
        list of int
            Control table values, concatenated for all IDs.
        """

        ids = makelist(ids)

        packet = [255, 255, 254, len(ids)+4, 132, addr, n] + ids + [0]
        packet[-1] = checksum(packet[2:-1])

        self.port.write(bytes(packet))
        response = list(self.port.read(6+len(ids)*n))

        data = []
        if len(response) == 6+len(ids)*n:
           if response[-1] == checksum(response[2:-1]):
              data = response[5:-1]
           else:
              print('Checksum error')
        elif len(response) == 0:
          print('No response')
        else:
          print('Incomplete response')

        return data

    def write(self, ids, addr, data):
        """Write multiple Dynamixel control table values.

        Parameters
        ----------
        ids : int or list of int
            Dynamixel IDs to write to.
        addr : int
            Start of control table entries to write.
        data : int
            Control table values to write, concatenated for all IDs.
        """

        ids = makelist(ids)

        n = int(len(data) / len(ids))
        packet = [255, 255, 254, len(
            data)+len(ids)+4, 131, addr, n] + [0]*(len(ids)+len(data)) + [0]
        for ii in range(len(ids)):
            packet[ii*(n+1)+7] = ids[ii]
            packet[ii*(n+1)+8:(ii+1)*(n+1)+7] = data[ii*n:(ii+1)*n]
        packet[-1] = checksum(packet[2:-1])

        self.port.write(bytes(packet))

    def getpos(self, ids):
        """Get current Dynamixel servo position(s).

        Parameters
        ----------
        ids : int or list of int
            Dynamixel IDs to read from.

        Returns
        -------
        list of int
            Positions in Dynamixel units.
        """
        ids = makelist(ids)
        n = len(ids)

        data = self.read(ids, 36, 2)

        pos = []
        if len(data) == n*2:
            pos = [None]*n
            for ii in range(n):
                raw = data[ii*2:(ii+1)*2]
                pos[ii] = raw[1]*256 + raw[0]

        errors = [ids[id] for (id, val) in enumerate(pos) if val == 65535]
        if len(errors) > 0:
            print('Error reading id(s) ' + str(errors))
            return []

        return pos

    def getangle(self, ids):
        """Get current Dynamixel servo angle(s).

        Parameters
        ----------
        ids : int or list of int
            Dynamixel IDs to read from.

        Returns
        -------
        list of float
            Angles in radians.
        """

        ids = makelist(ids)

        return [pos2rad(p) for p in self.getpos(ids)]

    def setpos(self, ids, pos, speed=None):
        """Set desired Dynamixel servo position(s).

        Parameters
        ----------
        ids : int or list of int
            Dynamixel IDs to write to.
        pos : int or list of int
            Desired position in Dynamixel units.
        speed : int or list of int, optional
            Movement speed in Dynamixel units.
        """

        ids = makelist(ids)
        n = len(ids)
        pos = makelist(pos, n)

        if speed is None:
            data = [0]*(n*2)
            for ii in range(n):
                raw = [pos[ii] & 255, pos[ii] >> 8]
                data[ii*2:(ii+1)*2] = raw
        else:
            speed = makelist(speed, n)

            data = [0]*(n*4)
            for ii in range(n):
                raw = [pos[ii] & 255, pos[ii] >> 8,
                       speed[ii] & 255, speed[ii] >> 8]
                data[ii*4:(ii+1)*4] = raw

        self.write(ids, 30, data)

    def setangle(self, ids, angle, speed=None):
        """Set desired Dynamixel servo angle(s).

        Parameters
        ----------
        ids : int or list of int
            Dynamixel IDs to write to.
        angle : float or list of float
            Desired position in radians.
        speed : float or list of float, optional
            Movement speed in radians per second.
        """

        ids = makelist(ids)
        n = len(ids)
        pos = [rad2pos(p) for p in makelist(angle, n)]
        if speed is not None:
            speed = [rads2speed(s) for s in makelist(speed, n)]

        return self.setpos(ids, pos, speed)

    def setspeed(self, ids, speed):
        """Set desired Dynamixel servo speeds.

        Parameters
        ----------
        ids : int or list of int
            Dynamixel IDs to write to.
        speed : float or list of float, optional
            Movement speed in radians per second.
        """

        angle = [copysign(2.1, s) for s in speed]
        speed = [abs(s) for s in speed]
        self.setangle(ids, angle, speed)

    def setcompliance(self, ids, compliance):
        """Set desired Dynamixel servo compliance slope(s).

        Parameters
        ----------
        ids : int or list of int
            Dynamixel IDs to write to.
        pos : int or list of int
            Desired compliance slope in Dynamixel units.
        """

        ids = makelist(ids)
        n = len(ids)
        compliance = [c for c in makelist(compliance, n) for _ in range(2)]

        self.write(ids, 28, compliance)

    def enable(self, ids):
        """Enable Dynamixel servo motor torque.
        
        Parameters
        ----------
        ids : int or list of int
            Dynamixel IDs to enable torque of.
        """

        ids = makelist(ids)

        self.write(ids, 24, [1]*len(ids))

    def disable(self, ids):
        """Disable Dynamixel servo motor torque.
        
        Parameters
        ----------
        ids : int or list of int
            Dynamixel IDs to disable torque of.
        """

        ids = makelist(ids)

        self.write(ids, 24, [0]*len(ids))

    def waitangle(self, ids):
        """Wait for current Dynamixel servo angle(s).

        Parameters
        ----------
        ids : int or list of int
            Dynamixel IDs to read from.

        Returns
        -------
        list of float
            Angles in radians.
        """
        ca = []
        while len(ca) == 0:
            ca = self.getangle(ids)
        return ca

    def move(self, ids, angle, speed=None, margin=0.05):
        """Move to desired Dynamixel servo angle(s).

        Parameters
        ----------
        ids : int or list of int
            Dynamixel IDs to write to.
        angle : float or list of float
            Desired position in radians.
        speed : float or list of float, optional
            Movement speed in radians per second.
        """
        ca = self.waitangle(ids)
        self.setangle(ids, angle, speed)
        while True:
            ca = self.waitangle(ids)
            ok = [abs(cur - des) < margin for (cur, des) in zip(ca, angle)]
            if all(ok):
                return


if __name__ == '__main__':
    arbotix = Arbotix('/dev/ttyUSB0')
    arbotix.setcompliance([1, 2, 3, 4, 5], 128)
    arbotix.setangle([1, 2, 3, 4, 5], 0, 0.5)
    while True:
        print(arbotix.getangle([1, 2, 3, 4, 5]))
