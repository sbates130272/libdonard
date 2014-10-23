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
##     Simple Tiff File parser
########################################################################

from __future__ import print_function

import sys
import os
import struct
import fractions
import numpy
import array

from PIL import Image


tiff_tags = {258: "BitsPerSample",
             320: "ColorMap",
             301: "ColorResponseCurves",
             259: "Compression",
             291: "GrayResponseCurve",
             290: "GrayResponseUnit",
             257: "ImageLength",
             256: "ImageWidth",
             254: "NewSubfileType",
             262: "PhotometricInterpretation",
             284: "PlanarConfiguration",
             317: "Predictor",
             296: "ResolutionUnit",
             278: "RowsPerStrip",
             277: "SamplesPerPixel",
             279: "StripByteCounts",
             273: "StripOffsets",
             282: "XResolution",
             283: "YResolution",
             315: "Artist",
             306: "DateTime",
             316: "HostComputer",
             270: "ImageDescription",
             271: "Make",
             272: "Model",
             305: "Software",
             292: "Group3Options",
             293: "Group4Options",
             269: "DocumentName",
             285: "PageName",
             297: "PageNumber",
             286: "XPosition",
             287: "YPosition",
             318: "WhitePoint",
             319: "PrimaryChromaticities",
             255: "SubfileType",
             274: "Orientation",
             263: "Threshholding",
             318: "ColorImageType",
             319: "ColorList",
             339: "SampleFormat",

}

typ_length = {1: 1,
              2: 1,
              3: 2,
              4: 4,
              5: 8}

def parse_tag(typ, data, length):
    if typ == 1:
        return struct.unpack("<%dB" % length, data)
    elif typ == 2:
        return data.decode("ascii").strip("\0")
    elif typ == 3:
        return struct.unpack("<%dH" % ((length + 1) // 2), data)
    elif typ == 4:
        return struct.unpack("<%dI" % ((length + 3) // 4), data)
    elif typ == 5:
        return fractions.Fraction(*struct.unpack("<II", data))


def unpack_file(fmt, f):
    sz = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(sz))

def read_tags(f):
    tags = []

    num_entries, = unpack_file("<H", f)
    for n in range(num_entries):
        tags.append(unpack_file("<HHII", f))
    next_ifd, = unpack_file("<I", f)

    tag_dict = {}
    for tag, typ, length, offset in tags:
        blength = typ_length[typ] * length

        if blength <= 4:
            data = struct.pack("<I", offset)[:blength]
        else:
            f.seek(offset)
            data = f.read(blength)

        if tag in tiff_tags:
            tag = tiff_tags[tag]


        value = parse_tag(typ, data, length)

        if tag in tag_dict:
            tag_dict[tag] = tuple(tag_dict[tag]) + tag
        elif len(value) == 1:
            tag_dict[tag] = value[0]
        else:
            tag_dict[tag] = value

    if next_ifd != 0:
        f.seek(next_ifd)
        tag_dict.update(read_tags(f))

    return tag_dict


def parse(f):
    id1, id2, ver, offset = unpack_file("<2cHI", f)
    idt = (id1 + id2).decode("ascii")

    print(f.name)
    print("ID: %s" % idt)
    assert idt == "II"
    print("Ver: %d" % ver)
    assert ver == 42
    print("Offset: %d" % offset)

    f.seek(offset)
    tags = read_tags(f)
    print()
    print("Tags: ")
    for t, v in tags.items():
        print("  %30s : %r" % (t, v))
    print()

    assert tags["Compression"] == 1
    assert tags["BitsPerSample"] == 32
    assert tags["SampleFormat"] == 3
    assert tags["PhotometricInterpretation"] == 1

    f.seek(tags["StripOffsets"])
    imdata = f.read(tags["StripByteCounts"])
    imdata = array.array("f", imdata)
    imdata = numpy.array(imdata)
    imdata = imdata.reshape((tags["ImageLength"], tags["ImageWidth"]),
                             order="C")

    print(imdata.shape)
    print(imdata)

    imdata *= 255
    im = Image.fromarray(imdata)
    im = im.convert("L")

    print(im.size, im.mode)
    im.save(os.path.splitext(f.name)[0] + ".jpg")



if __name__ == "__main__":
    for a in sys.argv[1:]:
        parse(open(a, "rb"))
