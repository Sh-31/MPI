#include <stdio.h>
#include "mpi.h"
int main(int argc, char* argv[]){

int rank, numOfProcesses, x, result;
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);

if(rank == 0) {x = 2;}

MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD); // will send x to all processes x = 2
x += rank; // x = 2 + rank if n = 4 then x = 2 + 0, 2 + 1, 2 + 2, 2 + 3 = 2, 3, 4, 5 

MPI_Reduce(&x, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);  // will sum all x values from all processes and store it in result at root process

// result = 2 + 3 + 4 + 5 = 14
if(rank == 0)
{
printf("Result = %i\n", result);
}
MPI_Finalize();
return 0;
}