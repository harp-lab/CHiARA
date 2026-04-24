#include "common.hpp"

int MPICH_Allreduce_ring(const char* sendbuf, char* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {


    int mpi_errno = MPI_SUCCESS;
    int i, src, dst;
    int nranks, is_inplace, rank;
    int extent;

    int *cnts, *displs;    /* Created for the allgatherv call */
    int send_rank, recv_rank, total_count;
    void *tmpbuf;
    MPI_Request reqs[2];      /* one send and one recv per transfer */

    is_inplace = (sendbuf == MPI_IN_PLACE);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    MPI_Type_size(datatype, &extent);

    cnts = (int*)malloc(nranks * sizeof(int));
    displs = (int*)malloc(nranks * sizeof(int));

    for(i = 0; i< nranks; i++){
        cnts[i] = 0;
    }

    total_count = 0;
    for (i = 0; i < nranks; i++) {
        cnts[i] = (count + nranks - 1) / nranks;
        if (total_count + cnts[i] > count) {
            cnts[i] = count - total_count;
            break;
        } else
            total_count += cnts[i];
    }

    displs[0] = 0;
    for (i = 1; i < nranks; i++)
        displs[i] = displs[i - 1] + cnts[i - 1];

    /* Phase 1: copy to tmp buf */
    if (!is_inplace) {
        memcpy(recvbuf, sendbuf, count * extent);
        //copied to recvbuf
    }

    tmpbuf = malloc(count * extent);
    
    if(!tmpbuf) {
        mpi_errno = MPI_ERR_OTHER;
        return mpi_errno;
    }

    src = (nranks + rank - 1) % nranks;
    dst = (rank + 1) % nranks;

    for (i = 0; i < nranks - 1; i++) {
        recv_rank = (nranks + rank - 2 - i) % nranks;
        send_rank = (nranks + rank - 1 - i) % nranks;

        mpi_errno = MPI_Irecv(tmpbuf, cnts[recv_rank], datatype, src, i, comm, &reqs[0]);
        if(mpi_errno!= MPI_SUCCESS) {
            free(tmpbuf);
            return mpi_errno;
        }
        mpi_errno = MPI_Isend(recvbuf+displs[send_rank] * extent, cnts[send_rank], datatype, dst, i, comm, &reqs[1]);
        
        if(mpi_errno!= MPI_SUCCESS) {
            free(tmpbuf);
            return mpi_errno;
        }
        mpi_errno = MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        if(mpi_errno!= MPI_SUCCESS) {
            free(tmpbuf);
            return mpi_errno;
        }
        mpi_errno = MPI_Reduce_local(tmpbuf, recvbuf + displs[recv_rank] * extent, cnts[recv_rank], datatype, op);
        if(mpi_errno!= MPI_SUCCESS) {
            free(tmpbuf);
            return mpi_errno;
        } 



    }

    mpi_errno = MPI_Allgatherv(MPI_IN_PLACE, -1, MPI_DATATYPE_NULL, recvbuf, cnts, displs, datatype, comm);
    if(mpi_errno!= MPI_SUCCESS) {
        free(tmpbuf);
        return mpi_errno;
    }


    free(tmpbuf);
    free(cnts);
    free(displs);

    return MPI_SUCCESS;


}

#ifdef DEBUG_MODE
int main(int argc, char** argv) {
    int rank, size;
    int count = 10; // Default number of elements to reduce
    double *sendbuf, *recvbuf, *expected, *mpi_result;
    int i, j, errors = 0;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Process command line arguments for count
    if (argc > 1) {
        count = atoi(argv[1]);
    }
    
    // Allocate buffers
    sendbuf = (double*)malloc(count * sizeof(double));
    recvbuf = (double*)malloc(count * sizeof(double));
    expected = (double*)malloc(count * sizeof(double));
    mpi_result = (double*)malloc(count * sizeof(double));
    
    if (!sendbuf || !recvbuf || !expected || !mpi_result) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Initialize send buffer with rank-specific data
    for (i = 0; i < count; i++) {
        sendbuf[i] = rank + i * 0.1;
    }
    
    // Calculate expected results (sum of all ranks for each element)
    for (i = 0; i < count; i++) {
        expected[i] = 0;
        for (j = 0; j < size; j++) {
            expected[i] += j + i * 0.1;
        }
    }
    
    if (rank == 0) {
        printf("Testing Ring Allreduce with:\n");
        printf("  Processes: %d\n", size);
        printf("  Elements per process: %d\n\n", count);
    }
    
    // For timing
    double start_time, end_time, elapsed_time, max_time;
    
    // Test our implementation
    memset(recvbuf, 0, count * sizeof(double));
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Call our custom allreduce implementation
    int rc = MPICH_Allreduce_ring(
        (char*)sendbuf, (char*)recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    // Get maximum time across all processes
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rc != MPI_SUCCESS) {
        if (rank == 0) {
            printf("Error: MPICH_Allreduce_ring returned %d\n", rc);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Verify results
    errors = 0;
    for (i = 0; i < count; i++) {
        if (fabs(recvbuf[i] - expected[i]) > 1e-10) {
            errors++;
            if (errors < 5) {  // Limit error output
                printf("Rank %d: Error at index %d: got %f, expected %f\n", 
                       rank, i, recvbuf[i], expected[i]);
            }
        }
    }
    
    // Collect error counts from all processes
    int total_errors;
    MPI_Reduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (total_errors == 0) {
            printf("Ring implementation: PASSED, Time: %.9f seconds\n", max_time);
        } else {
            printf("Ring implementation: FAILED with %d errors, Time: %.9f seconds\n", 
                   total_errors, max_time);
        }
    }
    
    // Compare with standard MPI_Allreduce
    memset(mpi_result, 0, count * sizeof(double));
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    MPI_Allreduce(sendbuf, mpi_result, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    // Get maximum time across all processes
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Standard MPI_Allreduce: Time: %.9f seconds\n", max_time);
    }
    
    // Test with different data sizes
    int test_sizes[] = {1, 100, 1000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int test = 0; test < num_tests; test++) {
        int test_count = test_sizes[test];
        
        // Skip if already tested this size
        if (test_count == count) continue;
        
        // Reallocate buffers for new size
        double *test_sendbuf = (double*)malloc(test_count * sizeof(double));
        double *test_recvbuf = (double*)malloc(test_count * sizeof(double));
        double *test_mpi_result = (double*)malloc(test_count * sizeof(double));
        
        if (!test_sendbuf || !test_recvbuf || !test_mpi_result) {
            fprintf(stderr, "Memory allocation failed for test size %d\n", test_count);
            continue;
        }
        
        // Initialize test data
        for (i = 0; i < test_count; i++) {
            test_sendbuf[i] = rank + i * 0.1;
        }
        
        // Run custom implementation
        rc = MPICH_Allreduce_ring((char*)test_sendbuf, (char*)test_recvbuf, 
                                test_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        if (rc != MPI_SUCCESS) {
            if (rank == 0) {
                printf("Size %d test: Error in MPICH_Allreduce_ring, rc=%d\n", 
                       test_count, rc);
            }
        } else {
            // Run standard MPI implementation
            MPI_Allreduce(test_sendbuf, test_mpi_result, test_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            // Compare results
            errors = 0;
            for (i = 0; i < test_count; i++) {
                if (fabs(test_recvbuf[i] - test_mpi_result[i]) > 1e-10) {
                    errors++;
                }
            }
            
            // Collect error counts
            MPI_Reduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            
            if (rank == 0) {
                if (total_errors == 0) {
                    printf("Size %d test PASSED\n", test_count);
                } else {
                    printf("Size %d test FAILED: %d errors\n", test_count, total_errors);
                }
            }
        }
        
        // Free test buffers
        free(test_sendbuf);
        free(test_recvbuf);
        free(test_mpi_result);
    }
    
    // Free memory
    free(sendbuf);
    free(recvbuf);
    free(expected);
    free(mpi_result);
    
    MPI_Finalize();
    return errors > 0 ? 1 : 0;
}


#endif