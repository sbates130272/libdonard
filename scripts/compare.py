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
##    Script to compare results of imgrep to answers provided by imhide
##
########################################################################

import re
import sys
import os

res_re = re.compile(r"^(?P<name>[^:]+):\s+"
                    r"(?P<x>[0-9]+)\+(?P<w>[0-9]+)\s+"
                    r"(?P<y>[0-9]+)\+(?P<h>[0-9]+)")

def parse_line(l):
    m = res_re.match(l)
    if not m: return None, None

    pos = tuple(int(m.group(n)) for n in ("x", "y", "w", "h"))

    name = os.path.splitext(m.group("name"))[0]
    name = os.path.abspath(name)

    return name, pos

def strip_name(name):
    name = name.replace("/mnt/princeton/", "")
    name = name.replace("/home/images/", "")
    return name

def load_answers(answers):
    results = {}

    adir = os.path.dirname(answers)
    for l in open(answers):
        name, pos = parse_line(l)
        if not name: continue
        name = strip_name(name)
        results.setdefault(name, set()).add(pos)

    return results

def find_match(pos, answers, threshold=1.5):
    x, y, w, h = pos
    threshold *= threshold

    for ax, ay, aw, ah in answers:
        if aw != w or ah != h:
            continue

        dist = (ax - x)**2 + (ay - y)**2
        if dist > threshold:
            continue

        answers.remove((ax, ay, aw, ah))
        return True

    return False

if __name__ == "__main__":
    import optparse
    import colour
    usage = "usage: %prog [options] ANSWER_KEY < RESULTS"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-t", "--threshold", action="store", type="float",
                      default=1.5, help="fuzz threshold for pixel locations")
    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.print_usage()
        sys.exit(-1)

    try:
        answers = load_answers(args[0])

        found = []
        false_positives = []

        unbuffered_stdin = os.fdopen(sys.stdin.fileno(), 'r', 0)

        while 1:
            l = unbuffered_stdin.readline()
            if not l: break

            print l,

            name, pos = parse_line(l)
            if not name: continue

            name = os.path.splitext(name)[0]
            name = strip_name(name)

            if name not in answers:
                false_positives.append(l)
                continue

            if not find_match(pos, answers[name], options.threshold):
                false_positives.append(l)
                continue

            if not answers[name]:
                del answers[name]

        print >> sys.stderr
        if not answers and not false_positives:
            print >> sys.stderr, colour.greenb("All Results Matched!")

        if false_positives:
            print >> sys.stderr, colour.redb("False Positives:")
            for fp in false_positives:
                print >> sys.stderr, fp,
            print >> sys.stderr

        if answers:
            print >> sys.stderr, colour.redb("Missed Matches:")
            for name, ans in answers.iteritems():
                for x,y,w,h in ans:
                    print >> sys.stderr, "%s: %5d+%3d %5d+%3d" % (name, x, w, y, h)

    except KeyboardInterrupt:
        pass
    except IOError, e:
        print e
