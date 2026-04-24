#include "common.hpp"
//WORKING
int MPICH_Allreduce_recursive_multiplying(const char* sendbuf, char* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,int k) {

    int mpi_errno = MPI_SUCCESS;
    int comm_size, rank, virt_rank;

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    virt_rank = rank;

    int pofk = 1;

    while(pofk*k <= comm_size) {
        pofk *= k;
    }

    char *tmp_buf;

    MPI_Request *reqs;

    int num_reqs = 0;

    reqs = (MPI_Request*) malloc(2 * (k - 1) * sizeof(MPI_Request));

    if(reqs == NULL) {
        return MPI_ERR_NO_MEM;
    }

    int typesize;
    MPI_Type_size(datatype, &typesize);
    int single_size = count * typesize;
    tmp_buf = (char*) malloc((k-1) * single_size);
    if(tmp_buf == NULL) {
        return MPI_ERR_NO_MEM;
    }

    if(sendbuf != MPI_IN_PLACE) {
        memcpy(recvbuf, sendbuf, single_size);
    }

    if(pofk < comm_size) {
        int is_commutative;
         MPI_Op_commutative(op, &is_commutative);
        if(!is_commutative) {
            //non pofk works only for commutative ops
            return MPI_ERR_OP;
        }

        if(rank >= pofk) {
            int pre_dst = rank % pofk;

            mpi_errno = MPI_Send(recvbuf,count, datatype, pre_dst, 0, comm);
            if(mpi_errno != MPI_SUCCESS) {
                return mpi_errno;
            }
            virt_rank = -1;
        }else{
            for(int pre_src = (rank % pofk) + pofk; pre_src < comm_size; pre_src += pofk){
                mpi_errno = MPI_Irecv(tmp_buf + num_reqs * single_size, count, datatype, pre_src, 0, comm, &reqs[num_reqs]);
                if(mpi_errno != MPI_SUCCESS) {
                    return mpi_errno;
                }
                num_reqs++;
            }

            MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);

            for(int i = 0; i < num_reqs; i++){
                if(i == (num_reqs - 1)){
                    mpi_errno = MPI_Reduce_local(tmp_buf + i * single_size, recvbuf, count, datatype, op);
                    if(mpi_errno!= MPI_SUCCESS) {
                        return mpi_errno;
                    }
                }else{
                    mpi_errno = MPI_Reduce_local(tmp_buf + i * single_size, tmp_buf + (i + 1) * single_size, count, datatype, op);
                    if(mpi_errno!= MPI_SUCCESS) {
                        return mpi_errno;
                    }
                }
            }
        }
    }

    if(virt_rank != -1) {
        num_reqs = 0;

        int exchanges = 0;
        int distance = 1;
        int next_distance = k;

        while(distance < pofk){
            int starting_rank = rank / next_distance * next_distance;
            int rank_offset = starting_rank + rank % distance;

            for(int dst = rank_offset; dst < starting_rank + next_distance; dst += distance ){
                if(dst != rank){
                    mpi_errno = MPI_Isend(recvbuf, count, datatype, dst, 0, comm, &reqs[num_reqs++]);
                    if(mpi_errno!= MPI_SUCCESS) {
                        return mpi_errno;
                    }
                    mpi_errno = MPI_Irecv(tmp_buf + exchanges * single_size, count, datatype, dst, 0, comm, &reqs[num_reqs++]);
                    if(mpi_errno!= MPI_SUCCESS) {
                        return mpi_errno;
                    }
                    exchanges++;
                }
                    
            }
        

            MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
            num_reqs = 0;
            exchanges = 0;

            int recvbuf_last = 0;

            for(int dst = rank_offset; dst < starting_rank + next_distance - distance; dst += distance ){
                char *dst_buf = tmp_buf + exchanges * single_size;

                if(dst == rank - distance){
                    mpi_errno = MPI_Reduce_local(dst_buf, recvbuf, count, datatype, op);
                    if(mpi_errno!= MPI_SUCCESS) {
                        return mpi_errno;
                    }
                    recvbuf_last = 1;
                    exchanges++;
                }else if(dst == rank){
                    mpi_errno = MPI_Reduce_local(recvbuf, dst_buf, count, datatype, op);
                    if(mpi_errno!= MPI_SUCCESS) {
                        return mpi_errno;
                    }
                    recvbuf_last = 0;
                }else{
                    mpi_errno = MPI_Reduce_local(dst_buf, dst_buf + single_size, count, datatype, op);
                    if(mpi_errno!= MPI_SUCCESS) {
                        return mpi_errno;
                    }
                    recvbuf_last = 0;
                    exchanges++;
                }
            }
            if(!recvbuf_last){
                memcpy(recvbuf, tmp_buf + (exchanges)*single_size, count*typesize);
            }
            distance = next_distance ;
            next_distance *= k;
            exchanges = 0;

        }
    }

    if(pofk < comm_size) {
        num_reqs = 0;
        if(rank >= pofk) {
            int post_src = rank % pofk;
            mpi_errno = MPI_Recv(recvbuf, count, datatype, post_src, 0, comm, MPI_STATUS_IGNORE);
            if(mpi_errno!= MPI_SUCCESS) {
                return mpi_errno;
            }
        }else{
            for(int post_dst = (rank % pofk) + pofk; post_dst < comm_size; post_dst += pofk){
                mpi_errno = MPI_Isend(recvbuf, count, datatype, post_dst, 0, comm, &reqs[num_reqs++]);
                if(mpi_errno!= MPI_SUCCESS) {
                    return mpi_errno;
                }
            }
            MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
        }
    }
    free(tmp_buf);
    free(reqs);

    return MPI_SUCCESS;

}

#ifdef DEBUG_MODE
int main(int argc, char** argv) {
    int rank, size;
    int count = 10; // Default number of elements to reduce
    double *sendbuf, *recvbuf, *expected;
    int i, j, errors = 0;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Process command line arguments for count and k
    if (argc > 1) {
        count = atoi(argv[1]);
    }
    
    // Default k value
    int default_k = 2;
    
    // Allocate buffers
    sendbuf = (double*)malloc(count * sizeof(double));
    recvbuf = (double*)malloc(count * sizeof(double));
    expected = (double*)malloc(count * sizeof(double));
    
    if (!sendbuf || !recvbuf || !expected) {
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
    
    // Test different k values
    int k_values[] = {2, 3, 4, 5 , 6, 7, 8, 9};
    int num_k_values = sizeof(k_values) / sizeof(k_values[0]);
    
    if (rank == 0) {
        printf("Testing Recursive Multiplying Allreduce with:\n");
        printf("  Processes: %d\n", size);
        printf("  Elements per process: %d\n", count);
        printf("  Testing k values: ");
        for (i = 0; i < num_k_values; i++) {
            printf("%d ", k_values[i]);
        }
        printf("\n\n");
    }
    
    // For timing
    double start_time, end_time, elapsed_time;
    double *times = (double*)malloc(num_k_values * sizeof(double));
    
    for (int k_idx = 0; k_idx < num_k_values; k_idx++) {
        int k = k_values[k_idx];
        
        // Skip if k is too large for the number of processes
        if (k > size) {
            if (rank == 0) {
                printf("Skipping k=%d (larger than process count %d)\n", k, size);
            }
            times[k_idx] = -1.0;
            continue;
        }
        
        // Reset recvbuf
        memset(recvbuf, 0, count * sizeof(double));
        
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        
        // Call our custom allreduce implementation with the current k value
        int rc = MPICH_Allreduce_recursive_multiplying(
            (char*)sendbuf, (char*)recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, k);
        
        end_time = MPI_Wtime();
        elapsed_time = end_time - start_time;
        
        // Get maximum time across all processes
        MPI_Reduce(&elapsed_time, &times[k_idx], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        if (rc != MPI_SUCCESS) {
            if (rank == 0) {
                printf("Error: MPICH_Allreduce_recursive_multiplying with k=%d returned %d\n", k, rc);
            }
            continue;
        }
        
        // Verify results
        errors = 0;
        for (i = 0; i < count; i++) {
            if (fabs(recvbuf[i] - expected[i]) > 1e-10) {
                errors++;
                if (rank == 0 && errors < 5) {  // Limit error output
                    printf("Error with k=%d at index %d: got %f, expected %f\n", 
                           k, i, recvbuf[i], expected[i]);
                }
            }
        }
        
        // Collect error counts from all processes
        int total_errors;
        MPI_Reduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (total_errors == 0) {
                printf("k=%d: PASSED, Time: %.9f seconds\n", k, times[k_idx]);
            } else {
                printf("k=%d: FAILED with %d errors, Time: %.9f seconds\n", 
                       k, total_errors, times[k_idx]);
            }
        }
    }
    
    // Compare performance of different k values
    if (rank == 0) {
        printf("\nPerformance Summary:\n");
        printf("------------------\n");
        for (i = 0; i < num_k_values; i++) {
            if (times[i] >= 0) {
                printf("k=%d: %.9f seconds", k_values[i], times[i]);
                
                // Find the best k value so far
                double best_time = times[0];
                int best_k_idx = 0;
                for (j = 1; j <= i; j++) {
                    if (times[j] >= 0 && times[j] < best_time) {
                        best_time = times[j];
                        best_k_idx = j;
                    }
                }
                
                if (i > 0) {
                    if (best_k_idx == i) {
                        printf(" (fastest so far)");
                    } else {
                        double speedup = times[best_k_idx] > 0 ? 
                            times[i] / times[best_k_idx] : 0;
                        printf(" (%.2fx slower than k=%d)", 
                               speedup, k_values[best_k_idx]);
                    }
                }
                printf("\n");
            }
        }
    }
    
    // Free memory
    free(sendbuf);
    free(recvbuf);
    free(expected);
    free(times);
    
    MPI_Finalize();
    return errors > 0 ? 1 : 0;
}

#endif // DEBUG_MODE