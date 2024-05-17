#include <mpi.h>

int main(int argc, char* argv[]) {
	int size;
	int rank;
	int number;
	int tag = 0;
	
	int left,right;
	int buffer[10], buffer2[10];
    MPI_Status status;


	// Initialize the MPI environment
    MPI_Init(&argc,&argv);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Get the rank of the process;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//Getting the right and left rank
	right = (rank + 1) % size;
    left = rank - 1;
    if (left < 0) left = size - 1;

	//Send to the left and receive from the right. (No deadlock will occur)
	MPI_Sendrecv(buffer, 10, MPI_INT, left, 123, buffer2,
				10, MPI_INT, right, 123, MPI_COMM_WORLD, &status);

    // Finalize the MPI environment.
    MPI_Finalize();

	return 0;
}