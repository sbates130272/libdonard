#!/usr/bin/env python
########################################################################
##
## Copyright 2014 PMC-Sierra, Inc.
##
## Licensed under the Apache License, Version 2.0 (the "License"); you
## may not use this file except in compliance with the License. You may
## obtain a copy of the License at
## http://www.apache.org/licenses/LICENSE-2.0 Unless required by
## applicable law or agreed to in writing, software distributed under the
## License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
## CONDITIONS OF ANY KIND, either express or implied. See the License for
## the specific language governing permissions and limitations under the
## License.
##
########################################################################

########################################################################
##
##   Author: Logan Gunthorpe
##
##   Date: Oct 23, 2014
##
##   Description:
##      Image grep test/example script
##
########################################################################

import os
import sys
import Image
import numpy as np
from numpy import fft

import utils
import time

data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

class ImageGrepError(Exception):
    pass

class ImageGrep(object):
    max_size  = 8192
    threshold = 150
    tiff      = False

    def __init__(self, needle, **kws):
        for k,v in kws.iteritems():
            if hasattr(self, k):
                setattr(self, k, v)

        data_needle = os.path.join(data, needle)
        if not os.path.exists(needle) and os.path.exists(data_needle):
            needle = data_needle

        im = Image.open(needle, 'r').convert("L")
        self.needle = np.array(im) / 255.
        self.needle_size = im.size

        revneedle = self.needle[::-1,::-1]
        revneedle = self._padimg(revneedle,
                                 self._next_highest_pow2(*revneedle.shape))

        edge_detect = np.ones((3,3)) * -1./8
        edge_detect[1,1] = 1
        edge_detect = self._padimg(edge_detect, revneedle.shape)
        edge_needle = self._convolve(revneedle, edge_detect)

        self.needle = self._padimg(edge_needle, (self.max_size,
                                                 self.max_size))

        self.pixels = 0
        self.bytes  = 0

    def _next_highest_pow2(self, *args):
        return tuple(1 << (x-1).bit_length() for x in args)

    def _padimg(self, a, shape):
        padspace = np.zeros(shape)
        padspace[:a.shape[0], :a.shape[1]] = a
        return padspace

    def _convolve(self, a, b):
        haystack_fft = fft.rfft2(a)
        needle_fft = fft.rfft2(b)

        return fft.irfft2((haystack_fft * needle_fft))

    def _save_image(self, m, fname):
        mx = np.amax(m)
        mn = np.amin(m)
        if mx  > 1.0 or mn < 0:
            m = m.copy()
            m += -mn
            m /= (mx-mn)

        Image.fromarray(np.uint8(m*255)).save(fname)

    def __call__(self, haystack, *args, **kws):

        if utils.istiff(haystack) and self.tiff:
            pass
        elif utils.isjpeg(haystack) and not self.tiff:
            pass
        else:
            return []

        im = Image.open(haystack, 'r').convert("L")
        self.pixels += im.size[0] * im.size[1]
        self.bytes  += os.path.getsize(haystack)

        haystack = np.array(im) / 255.

        #Pad dimensions to next highest power of 2 seeing this is
        # more efficient for the fft
        haystack = self._padimg(haystack, self._next_highest_pow2(*haystack.shape))

        if max(haystack.shape) > self.max_size:
            raise ImageGrepError("Image too large. Increase max_size.")

        needle = self.needle[:haystack.shape[0],:haystack.shape[1]]
        conv = self._convolve(needle, haystack)
        #self._save_image((conv > self.threshold) * 200, "conv.jpg")

        w, h = self.needle_size

        results = {}
        for x, y in zip(*np.nonzero(conv > self.threshold)):
            xx, yy = y-w+1, x-h
            if xx < 0 or yy < 0 or xx > im.size[0] or yy > im.size[1]:
                continue

            for (xr, yr, wr, hr), rr in results.iteritems():
                if (abs(xx-xr) < h or abs(yy-yr) < w):
                    if rr < conv[x,y]:
                        results[xx,yy,w,h] = conv[x,y]
                        del results[xr,yr,wr,hr]
                    break
            else:
                results[xx,yy,w,h] = conv[x, y]

        return results.items()

    def print_results(self, haystack, *args, **kws):
        try:
            for (x,y,w,h), r in self(haystack, *args, **kws):
                print "%s  %5d+%-3d %5d+%-3d    (%.2f)" % (haystack+":", x, w, y, h, r)
            sys.stdout.flush()
        except ImageGrepError, e:
            print >> sys.stderr, "%s (%s) " % (haystack, str(e))


if __name__ == "__main__":
    import optparse

    usage = "usage: %prog [options] IMAGE1 [IMAGE2 DIR1 ...]"
    parser = optparse.OptionParser(usage = usage)
    parser.add_option("--tiff", action="store_true",
                      help="process TIFF files rather than JPEGs")
    parser.add_option("-M", "--max-size", action="store", type="int",
                      default=ImageGrep.max_size,
                      help="maximum supported image size, default: %default")
    parser.add_option("-t", "--threshold", action="store", type="float",
                      default=ImageGrep.threshold,
                      help="detection threshold, default: %default")
    parser.add_option("-n", "--needle", action="store", type="string",
                      default="pmclogo.png", help="needle image to search for, default : %default")
    (options, args) = parser.parse_args()

    if not args:
        parser.print_usage()
        sys.exit(-1)
    imgrep = ImageGrep(**options.__dict__)

    try:
        starttime = time.time()
        utils.run(args, imgrep.print_results)
    except KeyboardInterrupt:
        pass
    except IOError, e:
        print e
    finally:
        duration = time.time() - starttime

        time.sleep(0.5)
        print >> sys.stderr
        print >> sys.stderr, ("%.2f%spixels in %.1fs    %.2f%spixels/s" %
                              (utils.si_suffix(imgrep.pixels) +
                               (duration, ) +
                               utils.si_suffix(imgrep.pixels / duration)))
        print >> sys.stderr, ("%.2f%sBytes in %.1fs    %.2f%sB/s" %
                              (utils.si_suffix(imgrep.bytes) +
                               (duration, ) +
                               utils.si_suffix(imgrep.bytes / duration)))
