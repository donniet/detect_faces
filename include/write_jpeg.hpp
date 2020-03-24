#pragma once


#include "jpeglib.h"
#include <stdio.h>
#include <vector>

#define BLOCK_SIZE 16384

void my_init_destination(j_compress_ptr cinfo)
{
  std::vector<JOCTET> * my_buffer = static_cast<std::vector<JOCTET>*>(cinfo->client_data);
  my_buffer->resize(BLOCK_SIZE);
  cinfo->dest->next_output_byte = &my_buffer->at(0);
  cinfo->dest->free_in_buffer = my_buffer->size();
}

boolean my_empty_output_buffer(j_compress_ptr cinfo)
{
  std::vector<JOCTET> * my_buffer = static_cast<std::vector<JOCTET>*>(cinfo->client_data);
  size_t oldsize = my_buffer->size();
  my_buffer->resize(oldsize + BLOCK_SIZE);
  cinfo->dest->next_output_byte = &my_buffer->at(oldsize);
  cinfo->dest->free_in_buffer = my_buffer->size() - oldsize;
  return true;
}

void my_term_destination(j_compress_ptr cinfo)
{
  std::vector<JOCTET> * my_buffer = static_cast<std::vector<JOCTET>*>(cinfo->client_data);
  my_buffer->resize(my_buffer->size() - cinfo->dest->free_in_buffer);
}
//
// cinfo->dest->init_destination = &my_init_destination;
// cinfo->dest->empty_output_buffer = &my_empty_output_buffer;
// cinfo->dest->term_destination = &my_term_destination;


void process_jpeg (JSAMPLE * image_buffer, int image_height, int image_width, int quality, std::vector<unsigned char>& out) {
  struct jpeg_compress_struct cinfo;
  /* This struct represents a JPEG error handler.  It is declared separately
  * because applications often want to supply a specialized error handler
  * (see the second half of this file for an example).  But here we just
  * take the easy way out and use the standard error handler, which will
  * print a message on stderr and call exit() if compression fails.
  * Note that this struct must live as long as the main JPEG parameter
  * struct, to avoid dangling-pointer problems.
  */
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer[1];	/* pointer to JSAMPLE row[s] */
  int row_stride;		/* physical row width in image buffer */

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  // make the output big enough to contain the entire output
  out.resize(3 * image_height * image_width);

  long unsigned int output_size = 0;
  unsigned char * output_data = NULL;

  jpeg_mem_dest(&cinfo, &output_data, &output_size);

  cinfo.image_width = image_width; 	/* image width and height, in pixels */
  cinfo.image_height = image_height;
  cinfo.input_components = 3;		/* # of color components per pixel */
  cinfo.in_color_space = JCS_RGB; 	/* colorspace of input image */

  std::vector<JOCTET> buffer(BLOCK_SIZE);

  cinfo.client_data = static_cast<void*>(&buffer);

  cinfo.dest->init_destination = &my_init_destination;
  cinfo.dest->empty_output_buffer = &my_empty_output_buffer;
  cinfo.dest->term_destination = &my_term_destination;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);
  jpeg_start_compress(&cinfo, TRUE);

  row_stride = image_width * 3;	/* JSAMPLEs per row in image_buffer */

  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = & image_buffer[cinfo.next_scanline * row_stride];
    (void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  out.assign(output_data, output_data + output_size);
}


void write_jpeg (JSAMPLE * image_buffer, int image_height, int image_width, const char * filename, int quality)
{
  struct jpeg_compress_struct cinfo;
  /* This struct represents a JPEG error handler.  It is declared separately
  * because applications often want to supply a specialized error handler
  * (see the second half of this file for an example).  But here we just
  * take the easy way out and use the standard error handler, which will
  * print a message on stderr and call exit() if compression fails.
  * Note that this struct must live as long as the main JPEG parameter
  * struct, to avoid dangling-pointer problems.
  */
  struct jpeg_error_mgr jerr;
  FILE * outfile;		/* target file */
  JSAMPROW row_pointer[1];	/* pointer to JSAMPLE row[s] */
  int row_stride;		/* physical row width in image buffer */

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  if ((outfile = fopen(filename, "wb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename);
    exit(1);
  }
  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = image_width; 	/* image width and height, in pixels */
  cinfo.image_height = image_height;
  cinfo.input_components = 3;		/* # of color components per pixel */
  cinfo.in_color_space = JCS_RGB; 	/* colorspace of input image */

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);
  jpeg_start_compress(&cinfo, TRUE);

  row_stride = image_width * 3;	/* JSAMPLEs per row in image_buffer */

  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = & image_buffer[cinfo.next_scanline * row_stride];
    (void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  jpeg_destroy_compress(&cinfo);
}
