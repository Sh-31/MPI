#include<stdio.h>
#include "mpi.h"

/*
 try :
    mpiexec -n 4  communicator
    mpiexec -n 8  communicator
    mpiexec -n 12 communicator
*/

int main()
{
    int myrank, comm_size;
     /*
     myrank -> rank of the process (id)
     comm_size -> total number of processes
     */
    MPI_Init(NULL, NULL);
    // MPI_COMM_WORLD -> Communicator
    /*
     communicator represents a communication domain which is essentially a set of processes that exchange messages between each other.
    */
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);  // This function gets the total number of processes in the communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); // This function gets the rank of the process calling it
    printf("Hello Procce %d from %d \n", myrank, comm_size);
    MPI_Finalize();
}