#include <stdio.h>
#include "mpi.h"
int main( int argc, char *argv [])
{
    int isend[3], irecv;
    int rank, size, i;

    MPI_Init ( &argc, &argv );
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
    MPI_Comm_size ( MPI_COMM_WORLD, &size );

    if (rank == 0) {
        for (i=0;i<size; i++) 
            isend[i] = i+1;
    }
    /*
    isend = [1,2,3] 
    */
    MPI_Scatter (&isend, 1, MPI_INT, &irecv, 1, MPI_INT, 0, MPI_COMM_WORLD) ;
    printf ("rank = %d: irecv = %d", rank, irecv); 
    /*
    (Its ok if the order is different)
    rank = 0: irecv = 1
    rank = 1: irecv = 2
    rank = 2: irecv = 3
    */
    MPI_Finalize();
    return 0;
}