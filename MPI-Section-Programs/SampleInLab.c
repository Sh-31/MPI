#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char* argv[]) {
	int size;int rank;int number;int tag = 0;
	int value = 100;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Request request;
	if (rank == 0)
	{
		MPI_Isend(&value, 1, MPI_INT, 1, 123, MPI_COMM_WORLD, &request);
	}

	else if (rank == 1)
	{
		MPI_Irecv(&value, 1, MPI_INT, 0, 123, MPI_COMM_WORLD, &request);
	}

	//I want to do a calculation
	int flag=0;
	for (int i = 0;i < 100000;i++)
	{
		int x = i + 1;

		if (flag == 0)
		{
			MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
			if(flag == 1)
				cout << "Rank " << rank << " Opertaion is done at " << i << endl;
		}
	}

	if (flag == 0)
	{
		MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
		if (flag == 1)
			cout << "Rank " << rank << " Opertaion is done at the end" << endl;
	} 

	MPI_Status status;
	MPI_Wait(&request, &status);
	
	if (flag == 0)
	{
		MPI_Test(&request, &flag, MPI_STATUS_IGNORE);

		if (flag == 1) cout << "Rank " << rank << " Opertaion is done at the very end" << endl;
	}

	MPI_Finalize();

	return 0;
}