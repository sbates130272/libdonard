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
##   Author: Stephen Bates <Stephen.Bates@pmcs.com>
##
##   Date: Oct 23, 2014
##
##   Description:
##    A module for generating and manipulating the image database used in
##    the Mt Donard program.
##
########################################################################

import os
import sys
import random
import Image

import utils

data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

class ImageHideError(Exception):
    pass

class ImageHideMaxReached(Exception):
    pass

class ImageHide(object):
    outdir  = "haystack"
    resize  = False
    tiff    = False
    skip    = 0.0
    maximum = 0
    count   = 0
    maxsize = 4096
    minsize = 1024
    maxpix  = 32768000

    def __init__(self, needle, **kws):
        for k,v in kws.iteritems():
            if hasattr(self, k):
                setattr(self, k, v)

        data_needle = os.path.join(data, needle)
        if not os.path.exists(needle) and os.path.exists(data_needle):
            needle = data_needle

        self.needle = Image.open(needle, 'r')

        answerkey = os.path.join(self.outdir, "answers.txt")
        self._ensure_directory(answerkey)
        self.fans = open(answerkey, "w")

    def _find_random_pos(self, img1, img2):
        """Find a random point in img1 to insert img2 whilst ensurimg img2
        is not cropped in any way."""

        if self._isbigger(img2, img1):
            raise ImageHideError("Needle image is larger than the haystack!")

        w1,h1 = img1.size
        w2,h2 = img2.size

        if w1==w2:
            x = 0;
        else:
            x = random.randrange(0,w1-w2)

        if h1==h2:
            y = 0
        else:
            y = random.randrange(0,h1-h2)

        return (x,y)

    def _randsize(self, img):
        """Determine a new size for an image using random."""

        w,h = img.size

        wn = random.randrange(1,w)
        hn = h*wn/w

        return (wn,hn)

    def _isbigger(self, img1, img2):
        """Returns True if img1 is bigger in either x or y direction, else
        returns False."""
        w1,h1 = img1.size
        w2,h2 = img2.size

        if w1>w2 or h1>h2:
            return True

        return False

    def _ensure_directory(self, fname):
        directory = os.path.dirname(fname)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _check_maxsize(self, img):
        w,h = img.size

        if w <= self.maxsize and h <= self.maxsize:
            return img

        scale = 4096. / max(w,h)
        newsize = (int(w*scale), int(h*scale))
        return img.resize(newsize)

    def _check_maxpix(self, img):
        w,h = img.size

        if w*h <= self.maxpix:
            return img

        scale = float(self.maxpix) / (w*h)
        newsize = (int(w*scale), int(h*scale))

        return img.resize(newsize)


    def _check_minsize(self, img):
        w,h = img.size

        if w >= self.minsize and h >= self.minsize:
            return img

        return None

    def _save_tiff(self, img, fname):
        img2 = img.convert("L")
        img2.save(fname)


    def __call__(self, haystack, reldir="."):
        """A function that takes a file that points to an image (haystack) and
        inserts into it a randomly sized and places image (as defined in
        self.needle). When resize is true we randomly vary the size of img2
        and ensure it fits in img1. When resize is false we return an
        error if img1 smaller than img2. The resultant is stored in an
        output file of the same name in self.outdir. Note that we
        sometimes skip the needle insertion."""

        if not utils.isjpeg(haystack):
            return []

        if self.maximum>0 and self.count>=self.maximum:
            raise ImageHideMaxReached("Reached image file limit.")

        img    = Image.open(haystack, 'r')

        if not self._check_minsize(img):
            return []

        img    = self._check_maxsize(img)
        img    = self._check_maxpix(img)
        needle = self.needle
        size   = self.needle.size

        # Now if resize if enabled we randomly resize the needle but
        # in such a way that we know it still fits inside the main image.
        if self.resize:
            needle = needle.copy()
            size = self._randsize(img)
            needle.resize(size, Image.ANTIALIAS)

        # Now insert needle into img and save in the results
        # folder. We should avoid overwriting either file as that will
        # lead to confusion of all kinds! Also make sure that we place
        # needle in such a way that the entire image is in the target
        # (i.e. no cropping).

        outf = os.path.relpath(haystack, reldir)
        outf.lstrip("./")
        outf = os.path.join(self.outdir, outf)
        if outf == haystack:
            raise ImageHideError("Won't overwrite the original file!")

        results = []

        if self.skip > random.random():
            results.append((outf, (-1,-1) + (-1,-1)))
        else:
            pos = self._find_random_pos(img, needle)
            img.paste(needle, pos, needle)
            results.append((outf, pos + size))

        self._ensure_directory(outf)
        img.convert("RGB").save(outf)
        if self.tiff:
            self._save_tiff(img, os.path.splitext(outf)[0]+".tiff")

        self.count = self.count + 1

        return results

    def print_results(self, fname_in, *args, **kws):
        try:
            a = self.fans
            for fname, (x, y, w, h) in self(fname_in, *args, **kws):
                fname = os.path.splitext(fname)[0]
                if all(v==-1 for v in (x,y,w,h)):
                    print "%s No Needle" % (fname+":")
                    print >> a, "%s No Needle" % (fname+":")
                else:
                    print "%s %5d+%-3d  %5d+%-3d" % (fname+":", x, w, y, h)
                    print >> a, "%s %5d+%-3d  %5d+%-3d" % (fname+":", x, w, y, h)
        except ImageHideError, e:
            fname_in = os.path.splitext(fname_in)[0]
            print >> sys.stderr, "%s (%s) " % (fname_in, str(e))

    def close(self):
        self.fans.close()

if __name__ == "__main__":
    import optparse

    usage = "usage: %prog [options] IMAGE1 [IMAGE2 DIR1 ...]"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-r", "--resize", action="store_true",
                      help="randomly resize the needle image")
    parser.add_option("-t", "--tiff", action="store_true",
                      help="store the output as a TIFF as well as a JPEG")
    parser.add_option("-n", "--needle", action="store", type="string",
                      default="pmclogo.png", help="the location of the needle image")
    parser.add_option("-s", "--skip", action="store", type="float",
                      default=0.00, help="the probability of skipping the needle insertion")
    parser.add_option("-m", "--maximum", action="store", type="int",
                      default=0, help="the maximum number of files processed (0 for all)")
    parser.add_option("-M", "--maxsize", action="store", type="int",
                      default=ImageHide.maxsize,
                      help="maximum size of images in either dimension"
                     " all images will be shrunk to meet this requirement. "
                     "default: %default")
    parser.add_option("-N", "--minsize", action="store", type="int",
                      default=ImageHide.minsize,
                      help="minimum size of images in either dimension"
                     " all images below this limit will be skipped. "
                     "default: %default")
    parser.add_option("-P", "--maxpix", action="store", type="int",
                      default=ImageHide.maxpix,
                      help="maximum filesize in pixels. When negative there is"
                      " no maximum. "
                     "default: %default")
    parser.add_option("-o", "--outdir", action="store", type="string",
                      default=ImageHide.outdir,
                      help="the location of the output directory, default: %default")
    (options, args) = parser.parse_args()

    if not args:
        parser.print_usage()
        sys.exit(-1)

    imhide = ImageHide(**options.__dict__)

    try:
        utils.run(args, imhide.print_results)

    except ImageHideMaxReached:
        pass
    except KeyboardInterrupt:
        pass
    except IOError, e:
        print e
