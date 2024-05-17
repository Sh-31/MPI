#include <stdio.h>
#include <string.h>
#include "mpi.h"



int main (int argc, char *argv[]) {

        int my_rank, p, tag=0;
        char msg [20];

        MPI_Status status;
        MPI_Init(&argc, &argv) ;
        MPI_Comm_size(MPI_COMM_WORLD, &p);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank) ;

    //      if (p < 2) {
    //     printf("This program requires at least 2 processes.\n");
    //     MPI_Abort(MPI_COMM_WORLD, 1);
    //     return 1;
    // }

        if (my_rank == 0) {

            sprintf_s(msg, "Hello p1");
            MPI_Send(msg, strlen(msg)+1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            
        }

        if (my_rank == 1){
             MPI_Recv(msg, 20, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
                printf("Message received: %s\n", msg);
                      
    }
        MPI_Finalize();
}