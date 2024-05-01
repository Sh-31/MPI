#include<stdio.h>
#include <string.h>
#include "mpi.h"


/* 
Point to point communication:  each process will communicate with another one.
    
    The code have to two main parts:
        1. Simple message passing from one process to another.
        2. use some of the MPI functions utilities an status object.

    NOTE:: MPI_Send and MPI_Recv are  blocking, asynchronous operations.
 */
const int MAX_STRING = 100;

int main()
{
    int myrank, comm_size;
    
    char massage[MAX_STRING];
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank !=0 ){
        sprintf_s(massage, "Hello from Procce %d from %d", myrank, comm_size);
        MPI_Send(massage, strlen(massage)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        /*
        MPI_Send(
            char* massage : message to send (void* buf)
            int count : length of the message
            MPI_Datatype : MPI_CHAR ,
            int dest, : rank of the receiver process,
            int tag : 0,
            MPI_Comm communicator : MPI_COMM_WORLD
        )

        Note: The (+1) is for (\n) character
           
        */
    }
    else {
            printf("Hello from Procce %d from %d \n", myrank, comm_size);
            for(int i = 1; i < comm_size; i++){
                MPI_Recv(massage, MAX_STRING, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                /*
                MPI_Recv(
                    char* massage : message to receive (void* buf)
                    int count : length of the message
                    MPI_Datatype : MPI_CHAR ,
                    int socuce : rank of the sender process,
                    int tag : 0,
                    MPI_Comm communicator : MPI_COMM_WORLD
                    // status : MPI_Status is optional
                )
                Note: 
                1. tage must be the same in both send and receive
                2. count IN MPI_Recv must be greater than the or equal length of the message
                */
                printf("%s\n", massage);
            }
    }

    MPI_Finalize();
}





// int main(){
//     // MPI_Recv operation
//     /* 
//             MPI_ANY_TAG : any tag
//             MPI_ANY_SOURCE : any source
    
//     */


//     int myrank, comm_size;
//     char massage[MAX_STRING];

//     MPI_Status status;
//     MPI_Init(NULL, NULL);
//     MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
//     if (myrank !=0 ){
//         sprintf_s(massage, "Hello from Procce %d from %d", myrank, comm_size);
//         MPI_Send(massage, strlen(massage)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
//     }
//     else {
//             printf("Hello from Procce %d from %d \n", myrank, comm_size);
//             for(int i = 1; i < comm_size; i++){
//                 MPI_Recv(massage, MAX_STRING, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
//                 printf("%s\n", massage);
//                 printf(" MPI_SOURCE :%d\n",status.MPI_SOURCE);
//                 printf(" MPI_TAG :%d\n",status.MPI_TAG);
//                 printf(" MPI_ERROR :%d\n",status.MPI_ERROR);
//             }
               
//     }



//     MPI_Finalize();
// }