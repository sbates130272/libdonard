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

#include "tiff.h"

#include <errno.h>

#include <endian.h>
#include <string.h>

#define EBAD_TIFF_IMAGE 0x800006

struct __attribute__ ((__packed__)) header {
    uint16_t byte_order;
    uint16_t version;
    uint32_t offset;
};

static uint16_t swap16(struct tiff_file *t, uint16_t x)
{
    if (t->byte_order == TIFF_LittleEndian)
        return htole16(x);
    else if (t->byte_order == TIFF_BigEndian)
        return htobe16(x);
    return x;
}

static uint32_t swap32(struct tiff_file *t, uint32_t x)
{
    if (t->byte_order == TIFF_LittleEndian)
        return htole32(x);
    else if (t->byte_order == TIFF_BigEndian)
        return htobe32(x);
    return x;
}

static int start_ifd(struct tiff_file *t, uint32_t offset)
{
    if (fseek(t->f, swap32(t, offset), SEEK_SET)) {
        errno = EBAD_TIFF_IMAGE;
        return -1;
    }

    if (fread(&t->cur_ifd_count, sizeof(t->cur_ifd_count), 1, t->f) != 1) {
        errno = EBAD_TIFF_IMAGE;
        return -1;
    }

    t->cur_ifd_count = swap16(t, t->cur_ifd_count);
    return 0;
}

struct tiff_file *tiff_init(FILE *f)
{
    struct header hdr;
    int ret;
    if ((ret = fread(&hdr, sizeof(hdr), 1, f)) != 1) {
        errno = EBAD_TIFF_IMAGE;
        return NULL;
    }

    if (!(hdr.byte_order == TIFF_LittleEndian &&
          hdr.version == htole16(TIFF_VERSION)) &&
        !(hdr.byte_order == TIFF_BigEndian &&
           hdr.version == htobe16(TIFF_VERSION)))
    {
        errno = EBAD_TIFF_IMAGE;
        return NULL;
    }

    struct tiff_file *t = malloc(sizeof(*t));
    if (t == NULL)
        return NULL;

    t->f = f;
    t->byte_order = hdr.byte_order;

    if (start_ifd(t, hdr.offset)) {
        free(t);
        return NULL;
    }


    return t;
}

struct tiff_file *tiff_open(const char *path)
{
    FILE *f = fopen(path, "r");
    if (f == NULL)
        return NULL;

    return tiff_init(f);
}

struct __attribute__ ((__packed__)) tag {
    uint16_t id;
    uint16_t type;
    uint32_t length;
    uint32_t offset;
};

int tag_lengths[] = {
    [TIFF_Byte] = 1,
    [TIFF_ASCII] = 1,
    [TIFF_Word] = 2,
    [TIFF_DWord] = 4,
    [TIFF_Rational] = 8,
};

static int read_offset_data(FILE *f, uint32_t offset, uint32_t blength,
                            void *dest)
{
    long orig = ftell(f);
    if (orig == -1)
        goto error_exit;

    if (fseek(f, offset, SEEK_SET))
        goto error_exit;

    if (fread(dest, blength, 1, f) != 1)
        goto error_exit;

    if (fseek(f, orig, SEEK_SET))
        goto error_exit;

    return 0;

error_exit:
    errno = EBAD_TIFF_IMAGE;
    return -1;
}

struct tiff_tag *tiff_read_tag(struct tiff_file *t)
{
    if (t->cur_ifd_count == 0) {
        uint32_t offset;
        if (fread(&offset, sizeof(offset), 1, t->f) != 1) {
            errno = EBAD_TIFF_IMAGE;
            return NULL;
        }

        if (offset == 0)
            return NULL;

        if (start_ifd(t, offset))
            return NULL;
    }

    t->cur_ifd_count--;

    struct tag ftag;
    if (fread(&ftag, sizeof(ftag), 1, t->f) != 1) {
        errno = EBAD_TIFF_IMAGE;
        return NULL;
    }

    uint16_t type = swap16(t, ftag.type);
    uint32_t count = swap32(t, ftag.length);
    uint32_t blength = tag_lengths[type] * count;

    if (type < 1 || type > 5) {
        errno = EBAD_TIFF_IMAGE;
        return NULL;
    }

    if (blength > 0xFFFF) {
        errno = EBAD_TIFF_IMAGE;
        return NULL;
    }

    struct tiff_tag *tag = malloc(sizeof(*tag) + blength);
    if (tag == NULL)
        return NULL;

    tag->id = swap16(t, ftag.id);
    tag->type = type;
    tag->count = count;
    tag->data.byte = (void *) &tag[1];

    if (blength <= 4) {
        memcpy(tag->data.byte, &ftag.offset, blength);
    } else {
        if (read_offset_data(t->f, swap32(t, ftag.offset), blength,
                             tag->data.byte))
        {
            free(tag);
            errno = EBAD_TIFF_IMAGE;
            return NULL;
        }
    }

    if (tag->type == TIFF_ASCII)
        tag->data.byte[blength-1] = 0;

    if (tag_lengths[tag->type] == 1)
        return tag;

    if (t->byte_order == TIFF_LittleEndian &&
        __BYTE_ORDER == __LITTLE_ENDIAN)
        return tag;

    if (t->byte_order == TIFF_BigEndian &&
        __BYTE_ORDER == __BIG_ENDIAN)
        return tag;

    if (type == TIFF_Word) {
        for(int i = 0; i < count; i++)
            tag->data.word[i] = __bswap_16(tag->data.word[i]);
    } else if (type == TIFF_DWord) {
        for(int i = 0; i < count; i++)
            tag->data.dword[i] = __builtin_bswap32(tag->data.dword[i]);
    } else if (type == TIFF_Rational) {
        for(int i = 0; i < count; i++) {
            tag->data.rational[i].num = __builtin_bswap32(tag->data.rational[i].num);
            tag->data.rational[i].den = __builtin_bswap32(tag->data.rational[i].den);
        }
    }

    return tag;
}

void tiff_free_tag(struct tiff_tag *tag)
{
    free(tag);
}


void tiff_close(struct tiff_file *t)
{
    fclose(t->f);
    free(t);
}

static const char *id_strings[] = {
    [TIFF_NewSubfileType] = "NewSubfileType",
    [TIFF_SubfileType] = "SubfileType",
    [TIFF_ImageWidth] = "ImageWidth",
    [TIFF_ImageLength] = "ImageLength",
    [TIFF_BitsPerSample] = "BitsPerSample",
    [TIFF_Compression] = "Compression",
    [TIFF_PhotometricInterpretation] = "PhotometricInterpretation",
    [TIFF_Threshholding] = "Threshholding",
    [TIFF_DocumentName] = "DocumentName",
    [TIFF_ImageDescription] = "ImageDescription",
    [TIFF_Make] = "Make",
    [TIFF_Model] = "Model",
    [TIFF_StripOffsets] = "StripOffsets",
    [TIFF_Orientation] = "Orientation",
    [TIFF_SamplesPerPixel] = "SamplesPerPixel",
    [TIFF_RowsPerStrip] = "RowsPerStrip",
    [TIFF_StripByteCounts] = "StripByteCounts",
    [TIFF_XResolution] = "XResolution",
    [TIFF_YResolution] = "YResolution",
    [TIFF_PlanarConfiguration] = "PlanarConfiguration",
    [TIFF_PageName] = "PageName",
    [TIFF_XPosition] = "XPosition",
    [TIFF_YPosition] = "YPosition",
    [TIFF_GrayResponseUnit] = "GrayResponseUnit",
    [TIFF_GrayResponseCurve] = "GrayResponseCurve",
    [TIFF_Group3Options] = "Group3Options",
    [TIFF_Group4Options] = "Group4Options",
    [TIFF_ResolutionUnit] = "ResolutionUnit",
    [TIFF_PageNumber] = "PageNumber",
    [TIFF_ColorResponseCurves] = "ColorResponseCurves",
    [TIFF_Software] = "Software",
    [TIFF_DateTime] = "DateTime",
    [TIFF_Artist] = "Artist",
    [TIFF_HostComputer] = "HostComputer",
    [TIFF_Predictor] = "Predictor",
    [TIFF_ColorImageType] = "ColorImageType",
    [TIFF_ColorList] = "ColorList",
    [TIFF_ColorMap] = "ColorMap",
    [TIFF_SampleFormat] = "SampleFormat",
};

const char *tiff_tag_name(enum tiff_tag_id id)
{
    return id_strings[id];
}
