#include<stdio.h>
#include <string.h>
#include "mpi.h"


const int MAX_STRING = 100;

void get_input(int my_rank, int comm_size, double* a_p , double* b_p, int* n_p);
void trap(double left_endpt, double right_endpt, int trap_count, double base_len);    

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

    }
    else {
            printf("Hello from Procce %d from %d \n", myrank, comm_size);
            for(int i = 1; i < comm_size; i++){
                MPI_Recv(massage, MAX_STRING, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("%s\n", massage);
            }
    }

    MPI_Finalize();
}

void get_input(int my_rank, int comm_size, double* a_p , double* b_p, int* n_p){

}