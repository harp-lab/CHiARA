
#include <mpi.h>
#include <iostream>
#include <cstring>   // for memcpy, memmove, etc.
#include <vector>
#include <fstream>
#include <cstdlib>



int MPICH_reduce_scatter_rec_halving(const char* sendbuf, char* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int MPICH_reduce_scatter_rec_doubling(const void *sendbuf, void *recvbuf, MPI_Aint recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int MPICH_reduce_scatter_radix(const void *sendbuf, void *recvbuf, MPI_Aint recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, int k);
int MPICH_reduce_scatter_pairwise(const void *sendbuf, void *recvbuf, MPI_Aint recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
                        