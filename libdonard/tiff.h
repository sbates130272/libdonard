////////////////////////////////////////////////////////////////////////
//
// Copyright 2014 PMC-Sierra, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you
// may not use this file except in compliance with the License. You may
// obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0 Unless required by
// applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for
// the specific language governing permissions and limitations under the
// License.
//
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
//
//   Author: Logan Gunthorpe
//
//   Date:   Oct 23 2014
//
//   Description:
//     TIFF Parsing Functions
//
////////////////////////////////////////////////////////////////////////


#ifndef __LIBDONARD_TIFF_H__
#define __LIBDONARD_TIFF_H__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define TIFF_VERSION 42

enum tiff_tag_id {
    TIFF_NewSubfileType = 254,
    TIFF_SubfileType = 255,
    TIFF_ImageWidth = 256,
    TIFF_ImageLength = 257,
    TIFF_BitsPerSample = 258,
    TIFF_Compression = 259,
    TIFF_PhotometricInterpretation = 262,
    TIFF_Threshholding = 263,
    TIFF_DocumentName = 269,
    TIFF_ImageDescription = 270,
    TIFF_Make = 271,
    TIFF_Model = 272,
    TIFF_StripOffsets = 273,
    TIFF_Orientation = 274,
    TIFF_SamplesPerPixel = 277,
    TIFF_RowsPerStrip = 278,
    TIFF_StripByteCounts = 279,
    TIFF_XResolution = 282,
    TIFF_YResolution = 283,
    TIFF_PlanarConfiguration = 284,
    TIFF_PageName = 285,
    TIFF_XPosition = 286,
    TIFF_YPosition = 287,
    TIFF_GrayResponseUnit = 290,
    TIFF_GrayResponseCurve = 291,
    TIFF_Group3Options = 292,
    TIFF_Group4Options = 293,
    TIFF_ResolutionUnit = 296,
    TIFF_PageNumber = 297,
    TIFF_ColorResponseCurves = 301,
    TIFF_Software = 305,
    TIFF_DateTime = 306,
    TIFF_Artist = 315,
    TIFF_HostComputer = 316,
    TIFF_Predictor = 317,
    TIFF_ColorImageType = 318,
    TIFF_ColorList = 319,
    TIFF_ColorMap = 320,
    TIFF_SampleFormat = 339,
};

enum tiff_field_type {
    TIFF_Byte = 1,
    TIFF_ASCII = 2,
    TIFF_Word = 3,
    TIFF_DWord = 4,
    TIFF_Rational = 5,
};

struct tiff_rational {
    uint32_t num;
    uint32_t den;
};

struct tiff_tag {
    enum tiff_tag_id id;
    enum tiff_field_type type;
    unsigned count;

    union {
        uint8_t *byte;
        char *ascii;
        uint16_t *word;
        uint32_t *dword;
        struct tiff_rational *rational;
    } data;
};

enum tiff_byte_order {
    TIFF_LittleEndian = 0x4949,
    TIFF_BigEndian = 0x4d4d,
};

struct tiff_file {
    FILE *f;
    enum tiff_byte_order byte_order;
    uint16_t cur_ifd_count;
};


#ifdef __cplusplus
extern "C" {
#endif

struct tiff_file *tiff_init(FILE *f);
struct tiff_file *tiff_open(const char *path);
struct tiff_tag *tiff_read_tag(struct tiff_file *t);
void tiff_free_tag(struct tiff_tag *tag);
void tiff_close(struct tiff_file *t);
const char *tiff_tag_name(enum tiff_tag_id id);

#ifdef __cplusplus
}
#endif

#endif
