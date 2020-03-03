#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call)                                                  \
{                                                                    \
   const cudaError_t error = call;                                   \ 
   if (error != cudaSuccess)                                         \
   {                                                                 \
      printf("Error: %s : %d",__FILE__,__LINE__);                    \
      printf("Code:%d,reason: %s\n",error,cudaGetErrorString(error));\
      exit(1);                                                       \
   }                                                                 \
}                                                                    \

void initiaData(float *ip, int size)
{
   time_t t;
   srand((unsigned int) time (&t));
   for (int j=0; j<size; j++)
   {
      ip[j] = (float)(rand() & 0xFF)/10.0f;
   }
}
double cpuSecond(){
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);

}

void checkResult(float *hostRef,float *gpuRef,const int N){
   double epsilon = 1.0E-8;
   int match = 1;
   for (int i=0; i<N; i++){
      if(abs(hostRef[i] - gpuRef[i]) > epsilon){
         match = 0;
         printf("Arrays do not match \n");
         printf("host %5.2f gpu %5.2f at current %d \n",hostRef[i],gpuRef[i],i);
         break; 
      }
   }
   if (match) printf("Arrays match\n");
   return;
}


void sumMatrixOnHost(float *A,float *B,float *C,const int nx,const int ny){
   float *ia = A;
   float *ib = B;
   float *ic = C;
   
   for (int iy = 0;iy < ny; iy++){
      for (int ix = 0; ix < nx; ix++){
         ic[ix] = ia[ix] + ib[ix];
      }
      ia += nx; ib += nx; ic += nx;
   }
   
}

__global__ void sumMatrixOnGPU2D(float *MatA,float *MatB,float *MatC,const int nx,const int ny){
   unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
   unsigned int idx = iy*nx + ix;
   if (idx == 0)
      printf("blockIdx.x = %d",blockIdx.x); 
   if (ix < nx && iy < ny){
      MatC[idx] = MatA[idx] + MatB[idx];

   }
}


int main(int argc,char **argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using Device %d:%s\n",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));
    
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx * ny;
    printf("Matrix size: %d %d\n",nx,ny);
    int nBytes = nxy * sizeof(float);
 
    float *h_A,*h_B,*hostRef,*gpuRef;
   
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    
    double iStart,iElaps;
    iStart = cpuSecond();  
    initiaData(h_A,nxy);
    initiaData(h_B,nxy);
    iElaps = cpuSecond() - iStart;

    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);
    
    iStart = cpuSecond();
    sumMatrixOnHost(h_A,h_B,hostRef,nx,ny);
    iElaps = cpuSecond() - iStart;

    float *d_A,*d_B,*d_C;
    cudaMalloc((void **)&d_A,nBytes);
    cudaMalloc((void **)&d_C,nBytes);
    cudaMalloc((void **)&d_B,nBytes);
   
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);
   
    dim3 block(32,32);
    dim3 grid ((nx + block.x -1)/block.x,(ny + block.y - 1)/block.y);
    iStart = cpuSecond();
    sumMatrixOnGPU2D <<< grid,block >>>(d_A,d_B,d_C,nx,ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPU2D <<< (%d,%d),(%d,%d) >>> elapsed %f sec\n",block.x,block.y,grid.x,grid.y,iElaps);
   
    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
    
    checkResult(hostRef,gpuRef,nxy);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

   return 0;
 
}
