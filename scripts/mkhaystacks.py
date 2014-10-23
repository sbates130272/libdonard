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
##    A script that usese the imhide class to remove old haystacks and
##    create new ones.
##
########################################################################

import os
import sys
import random
import shutil

import imhide
import utils
from collections import namedtuple

  # Declare some defaults for directory locations, these can be
  # overwritten by inputs (TBD).

szDatabase     = '/home/images/SUN397/'
szHaystacksHdd = '/home/images/'
szHaystacksSsd = '/mnt/princeton/'
szNeedle       = '/home/images/pmclogo.png'

  # Declare a tuple list of the properties of each of the haystacks to
  # be created. This provides the parameters for each in terms of the
  # number of images and the probability of needles in each

ntHaystack = namedtuple('ntHaystack',
                        'name maximum skip')

liHaystacks = [ ntHaystack('tinydb',   10, 0.75)   ,
                ntHaystack('smalldb',  100, 0.9)   ,
                ntHaystack('mediumdb', 1000, 0.9)  ,
                ntHaystack('largedb', 10000, 0.99) ,
                ntHaystack('hugedb',  50000, 0.999),
               ]

if __name__ == "__main__":
    import optparse

    parser = optparse.OptionParser()
    parser.add_option("-f", "--force", action="store_true",
                      help="do not prompt user, just delete and create new databases")
    parser.add_option("-r", "--resize", action="store_true",
                      help="randomly resize the needle image")
    (options, args) = parser.parse_args()

    if not options.force:
        var = raw_input("Warning this will remove any existing "
        "haystacks; type ""Y"" ENTER to continue. ")
        if var != "Y":
            exit(-1)

    options.tiff   = True
    options.needle = szNeedle

    for haystack in liHaystacks:
        print haystack.name

        if os.path.exists(szHaystacksHdd+haystack.name):
            shutil.rmtree(szHaystacksHdd+haystack.name)

        options.outdir  = szHaystacksHdd+haystack.name
        options.maximum = haystack.maximum
        options.skip    = haystack.skip

        _imhide = imhide.ImageHide(**options.__dict__)

        try:
            utils.run([szDatabase], _imhide.print_results)
        except imhide.ImageHideMaxReached:
            pass
        except KeyboardInterrupt:
            pass
        except IOError, e:
            print e

        _imhide.close()

        if os.path.exists(szHaystacksSsd+haystack.name):
            shutil.rmtree(szHaystacksSsd+haystack.name)

        shutil.copytree(szHaystacksHdd+haystack.name,
                        szHaystacksSsd+haystack.name)
