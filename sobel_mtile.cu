#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "string.h"
#include "cuda.h"
//#include<fstream>
//#include<iostream>

// This code is written by Arash Nemati Hayati 00763261 - Feb 23, 2016
#define opt 2 // The version to be used

#define DEFAULT_THRESHOLD  4000
#define DEFAULT_FILENAME "BWstop-sign.ppm"

#define BLOCK_SIZE 32 // Number of threads in x and y direction - Maximum Number of threads per block = 32 * 32 = 1024
#define TILE_SIZE 32 // Tile size for memory hierarchy optimizations
#define version_1 1 // Using global memory
#define version_2 2 // Using memory hierarchy optimizations


__global__ void sobel( int xd_size, int yd_size, int maxdval, int d_thresh, unsigned int *input , int *output)
{
	
// Version 1 - Using global memory
	
	if (opt == 1){	
        int magnitude, sum1, sum2;
        int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
        int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
	
        if ((i < yd_size) && (j < xd_size))
        {
        output[i * xd_size + j] = 0;
        }
	__syncthreads();

        if ((i > 0) && (i < yd_size - 1) && (j > 0) && (j < xd_size - 1))
        {
        int offset = i * xd_size + j; 
	
        sum1 =  input[ xd_size * (i-1) + j+1 ] -     input[ xd_size*(i-1) + j-1 ]
        + 2 * input[ xd_size * (i)   + j+1 ] - 2 * input[ xd_size*(i)   + j-1 ]
        +     input[ xd_size * (i+1) + j+1 ] -     input[ xd_size*(i+1) + j-1 ];

        sum2 = input[ xd_size * (i-1) + j-1 ] + 2 * input[ xd_size * (i-1) + j ]  + input[ xd_size * (i-1) + j+1 ]
            - input[xd_size * (i+1) + j-1 ] - 2 * input[ xd_size * (i+1) + j ] - input[ xd_size * (i+1) + j+1 ];

        magnitude =  sum1*sum1 + sum2*sum2;

      if (magnitude > d_thresh){
        output[offset] = 255;
        }
      else{
        output[offset] = 0;
        }
    }
	__syncthreads();
}

// Version 2 - Using memory hierarchy optimization
	
	if (opt == 2){
	int magnitude, sum1, sum2;
	int tile_width = TILE_SIZE;
	__shared__ int temp[TILE_SIZE * TILE_SIZE];
	
	int bx = blockIdx.x; int by = blockIdx.y; 
	int tid_x = threadIdx.x; int tid_y = threadIdx.y;
	int row = by * tile_width + tid_y; // row
	int col = bx * tile_width + tid_x; // col

        if ((row < yd_size) && (col < xd_size))
        {
        output[row * xd_size + col] = 0;
        }
        __syncthreads();
	
	temp[tid_x * tile_width + tid_y]  = 0; // Initializae the temp tile matrix which stores a block chunk of data
	__syncthreads();
	
	for (int m = 0; m < xd_size / tile_width; ++m){
	temp[tid_y * tile_width + tid_x] = input[row * xd_size + (m * tile_width + tid_x)];
	
//	temp[tid_x * tile_width + tid_y] = input[i * xd_size + j];
	__syncthreads();

	if ((tid_x > 0) && (tid_x < tile_width - 1) && (tid_y > 0) && (tid_y < tile_width - 1))
        {
        int offset = row * xd_size + col;

        sum1 =  temp[ tile_width * (tid_y-1) + tid_x+1 ] -     temp[ tile_width*(tid_y-1) + tid_x-1 ]
        + 2 * temp[ tile_width * (tid_y)   + tid_x+1 ] - 2 * temp[ tile_width*(tid_y)   + tid_x-1 ]
        +     temp[ tile_width * (tid_y+1) + tid_x+1 ] -     temp[ tile_width*(tid_y+1) + tid_x-1 ];

        sum2 = temp[ tile_width * (tid_y-1) + tid_x-1 ] + 2 * temp[ tile_width * (tid_y-1) + tid_x ]  + temp[ tile_width * (tid_y-1) + tid_x+1 ]
            - temp[tile_width * (tid_y+1) + tid_x-1 ] - 2 * temp[ tile_width * (tid_y+1) + tid_x ] - temp[ tile_width * (tid_y+1) + tid_x+1 ];

//	if (tid_x > 0) && (tid_y == 0){
//	}
	

        magnitude =  sum1*sum1 + sum2*sum2;

      if (magnitude > d_thresh){
        output[offset] = 255;
        }
      else{
        output[offset] = 0;
        }
	}
	__syncthreads();
    }
}	
}

unsigned int *read_ppm( char *filename, int & xsize, int & ysize, int & maxval ){
  
  if ( !filename || filename[0] == '\0') {
    fprintf(stderr, "read_ppm but no file name\n");
    return NULL;  // fail
  }

  fprintf(stderr, "read_ppm( %s )\n", filename);
  int fd = open( filename, O_RDONLY);
  if (fd == -1) 
    {
      fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
      return NULL; // fail 

    }

  char chars[1024];
  int num = read(fd, chars, 1000);

  if (chars[0] != 'P' || chars[1] != '6') 
    {
      fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
      return NULL;
    }

  unsigned int width, height, maxvalue;


  char *ptr = chars+3; // P 6 newline
  if (*ptr == '#') // comment line! 
    {
      ptr = 1 + strstr(ptr, "\n");
    }

  num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
  fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
  xsize = width;
  ysize = height;
  maxval = maxvalue;
  
  unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
  if (!pic) {
    fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
    return NULL; // fail but return
  }

  // allocate buffer to read the rest of the file into
  int bufsize =  3 * width * height * sizeof(unsigned char);
  if (maxval > 255) bufsize *= 2;
  unsigned char *buf = (unsigned char *)malloc( bufsize );
  if (!buf) {
    fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
    return NULL; // fail but return
  }





  // TODO really read
  char duh[80];
  char *line = chars;

  // find the start of the pixel data.   no doubt stupid
  sprintf(duh, "%d\0", xsize);
  line = strstr(line, duh);
  //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
  line += strlen(duh) + 1;

  sprintf(duh, "%d\0", ysize);
  line = strstr(line, duh);
  //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
  line += strlen(duh) + 1;

  sprintf(duh, "%d\0", maxval);
  line = strstr(line, duh);


  fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
  line += strlen(duh) + 1;

  long offset = line - chars;
  lseek(fd, offset, SEEK_SET); // move to the correct offset
  long numread = read(fd, buf, bufsize);
  fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize); 

  close(fd);


  int pixels = xsize * ysize;
  for (int i=0; i<pixels; i++) pic[i] = (int) buf[3*i];  // red channel

 

  return pic; // success
}

void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) 
{
  FILE *fp;
  
  fp = fopen(filename, "w");
  if (!fp) 
    {
      fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
      exit(-1); 
    }
  int x,y;
  
  
  fprintf(fp, "P6\n"); 
  fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
  
  int numpix = xsize * ysize;
  for (int i=0; i<numpix; i++) {
    unsigned char uc = (unsigned char) pic[i];
    fprintf(fp, "%c%c%c", uc, uc, uc); 
  }
  fclose(fp);

}

/*
__global__ void sobel( int xd_size, int yd_size, int maxdval, int d_thresh, int *d_result , unsigned int *d_pic)
{
	int magnitude, sum1, sum2;
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
//	int numbytes = xd_size * yd_size * 3 * sizeof(int);
	
	int *out = d_result;	
	if ((j < yd_size) && (i < xd_size))
	{			
	*out++ = 0;
	}
	  
	if ((j > 0) && (j < yd_size - 1) && (i > 0) && (i < xd_size - 1))
	{
	int offset = i*xd_size + j;

        sum1 =  d_pic[ xd_size * (i-1) + j+1 ] -     d_pic[ xd_size*(i-1) + j-1 ]
        + 2 * d_pic[ xd_size * (i)   + j+1 ] - 2 * d_pic[ xd_size*(i)   + j-1 ]
        +     d_pic[ xd_size * (i+1) + j+1 ] -     d_pic[ xd_size*(i+1) + j-1 ];

        sum2 = d_pic[ xd_size * (i-1) + j-1 ] + 2 * d_pic[ xd_size * (i-1) + j ]  + d_pic[ xd_size * (i-1) + j+1 ]
            - d_pic[xd_size * (i+1) + j-1 ] - 2 * d_pic[ xd_size * (i+1) + j ] - d_pic[ xd_size * (i+1) + j+1 ];

        magnitude =  sum1*sum1 + sum2*sum2;

      if (magnitude > d_thresh){
	d_result[offset] = 255;
	}
      else{
        d_result[offset] = 0;
	}
    }
}*/
	
main( int argc, char **argv )
{

  int thresh = DEFAULT_THRESHOLD;
  char *filename;
  cudaError_t error;
  filename = strdup( DEFAULT_FILENAME);
  
  if (argc > 1) {
    if (argc == 3)  { // filename AND threshold
      filename = strdup( argv[1]);
       thresh = atoi( argv[2] );
    }
    if (argc == 2) { // default file but specified threshhold
      
      thresh = atoi( argv[1] );
    }

    fprintf(stderr, "file %s    threshold %d\n", filename, thresh); 
  }


  int xsize, ysize, maxval;
 // int dev_xsize, dev_ysize, dev_maxval;  
  unsigned int *pic = read_ppm( filename, xsize, ysize, maxval ); 
  unsigned int *dev_pic;

  int numbytes =  xsize * ysize * 1 * sizeof( int );
  int *result = (int *) malloc( numbytes );
  int *dev_result;
 
  cudaMalloc ((void **)&dev_result, numbytes);
  cudaMalloc ((void **)&dev_pic, numbytes);
  error = cudaMalloc ((void **) &dev_pic, numbytes);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (devic,d) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
  cudaMemcpy( dev_pic, pic, numbytes, cudaMemcpyHostToDevice);

  if (!result) { 
    fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
    exit(-1); // fail
  }

int gridsize_x = xsize/BLOCK_SIZE + 1;
	int gridsize_y = ysize/BLOCK_SIZE + 1;
        dim3 dimgrid(gridsize_x, gridsize_y, 1); // The grid has #gridsize blocks in x and 1 block in y and 1 block in z direction
        dim3 dimblock(BLOCK_SIZE, BLOCK_SIZE, 1);

	fprintf(stderr,".....Number of block in x dir......%d\n",gridsize_x);
	fprintf(stderr,".....Number of block in y dir......%d\n",gridsize_y);
	fprintf(stderr,".....Number of threads in x dir....%d\n",BLOCK_SIZE);
        fprintf(stderr,".....Number of threads in y dir....%d\n",BLOCK_SIZE);	
        fprintf(stderr,".....xsize (Number of columns).....%d\n",xsize);
        fprintf(stderr,".....ysize (Number of rows)........%d\n",ysize,"\n");

  // Initialize timer
  cudaEvent_t start,stop;
  float elapsed_time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  sobel<<<dimgrid, dimblock>>>(xsize, ysize, maxval, thresh, dev_pic, dev_result);
  
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time,start, stop);
  cudaMemcpy( result, dev_result, numbytes, cudaMemcpyDeviceToHost);
 
/*  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time,start, stop);
 */
  write_ppm( "result.ppm", xsize, ysize, 255, result);

  fprintf(stderr, "sobel done\n"); 

 // fprintf(stderr,".....output[0][0]....%d\n",result[0]);
 
  printf("The operation was successful, time = %2.6f %s\n", elapsed_time, "ms");

  cudaFree(dev_result);
  free(result);
}

