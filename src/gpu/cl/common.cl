#define DEVICE
#define GLOBAL __global
#define KERNEL __kernel
#define LOCAL __local
#define CONSTANT __constant

#define GET_GLOBAL_ID() get_global_id(0)
#define GET_GROUP_ID() get_group_id(0)
#define GET_LOCAL_ID() get_local_id(0)
#define GET_LOCAL_SIZE() get_local_size(0)
#define BARRIER_LOCAL() barrier(CLK_LOCAL_MEM_FENCE)

#ifdef __NV_CL_C_VERSION
#define OPENCL_NVIDIA
#endif
