#include "common.hpp"

//WORKING
int MPICH_Allreduce_recursive_doubling(const char* sendbuf, char* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

    int comm_size, rank, mpi_errno = MPI_SUCCESS, mask, dst, is_commutative, pof2, newrank, rem, newdst, typesize;

    char * tmp_buf = NULL;


    MPI_Op_commutative(op, &is_commutative);



    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);
    MPI_Type_size(datatype, &typesize);
    tmp_buf = (char*)malloc(count * typesize);

    if(sendbuf != MPI_IN_PLACE) {
        memcpy(recvbuf, sendbuf, count * typesize);
    }

    //get nearest power of 2 less than or equal to comm_size
    pof2 = 1;
    while(pof2 <= comm_size) {
        pof2 <<= 1;
    }

    pof2 >>= 1;

    rem = comm_size - pof2;

    if(rank < 2 * rem) {
        if(rank % 2 == 0) {
            mpi_errno = MPI_Send(recvbuf, count, datatype, rank + 1, 0, comm);

            if(mpi_errno != MPI_SUCCESS) {
                return mpi_errno;
            }
            newrank = -1;
        }else{
            mpi_errno = MPI_Recv(tmp_buf, count, datatype, rank - 1, 0, comm, MPI_STATUS_IGNORE);

            if(mpi_errno!= MPI_SUCCESS) {
                return mpi_errno;
            }
            mpi_errno = MPI_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
            if(mpi_errno!= MPI_SUCCESS) {
                return mpi_errno;
            }
            newrank = rank / 2;
        }
    }else{
        newrank = rank - rem;
    }

    if(newrank != -1) {
        mask = 0x1;
        while(mask < pof2) {
            newdst = newrank ^ mask;
            dst = (newdst < rem) ? (newdst << 1) + 1 : newdst + rem;

            mpi_errno = MPI_Sendrecv(recvbuf, count, datatype, dst, 0, tmp_buf, count, datatype, dst, 0, comm, MPI_STATUS_IGNORE);
            if(mpi_errno!= MPI_SUCCESS) {
                return mpi_errno;
            }

            if(is_commutative || (dst<rank)){
                mpi_errno = MPI_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
                if(mpi_errno!= MPI_SUCCESS) {
                    return mpi_errno;
                }
            } else {

                mpi_errno = MPI_Reduce_local(recvbuf, tmp_buf, count, datatype, op);
                if(mpi_errno!= MPI_SUCCESS) {
                    return mpi_errno;
                }
                memcpy(recvbuf, tmp_buf, count * typesize);
            }
            mask <<= 1;
        }
    }

    if (rank < 2 * rem) {
        if(rank % 2) {

            mpi_errno = MPI_Send(recvbuf, count, datatype, rank - 1, 0, comm);
            
        }else{
            mpi_errno = MPI_Recv(recvbuf, count, datatype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
        }
        if(mpi_errno!= MPI_SUCCESS) {
            return mpi_errno;
        }
    }

    free(tmp_buf);
    return MPI_SUCCESS;

}

#ifdef DEBUG_MODE
int main(int argc, char** argv) {
    int rank, size;
    int count = 10; // Number of elements to reduce
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
    
    // Call our custom allreduce implementation
    MPICH_Allreduce_recursive_doubling((char*)sendbuf, (char*)recvbuf, 
                                       count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // Verify results
    errors = 0;
    for (i = 0; i < count; i++) {
        if (fabs(recvbuf[i] - expected[i]) > 1e-10) {
            errors++;
            if (rank == 0) {
                printf("Error at index %d: got %f, expected %f\n", 
                       i, recvbuf[i], expected[i]);
            }
        }
    }
    
    // Report results
    if (rank == 0) {
        if (errors == 0) {
            printf("Test PASSED: All %d values match expected results\n", count);
        } else {
            printf("Test FAILED: %d of %d values don't match expected results\n", 
                   errors, count);
        }
    }
    
    // Compare with standard MPI_Allreduce for validation
    MPI_Allreduce(sendbuf, mpi_result, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    errors = 0;
    for (i = 0; i < count; i++) {
        if (fabs(recvbuf[i] - mpi_result[i]) > 1e-10) {
            errors++;
            if (rank == 0) {
                printf("Mismatch with MPI_Allreduce at index %d: custom %f, MPI %f\n", 
                       i, recvbuf[i], mpi_result[i]);
            }
        }
    }
    
    if (rank == 0) {
        if (errors == 0) {
            printf("Validation PASSED: Results match standard MPI_Allreduce\n");
        } else {
            printf("Validation FAILED: %d of %d values don't match MPI_Allreduce\n", 
                   errors, count);
        }
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
        MPICH_Allreduce_recursive_doubling((char*)test_sendbuf, (char*)test_recvbuf, 
                                           test_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Run standard MPI implementation
        MPI_Allreduce(test_sendbuf, test_mpi_result, test_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Compare results
        errors = 0;
        for (i = 0; i < test_count; i++) {
            if (fabs(test_recvbuf[i] - test_mpi_result[i]) > 1e-10) {
                errors++;
            }
        }
        
        if (rank == 0) {
            if (errors == 0) {
                printf("Size %d test PASSED\n", test_count);
            } else {
                printf("Size %d test FAILED: %d of %d values don't match\n", 
                       test_count, errors, test_count);
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
#endif // DEBUG_MODE

