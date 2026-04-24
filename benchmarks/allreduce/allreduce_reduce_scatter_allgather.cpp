#include "common.hpp"

int MPICH_Allreduce_reduce_scatter_allgather(const char* sendbuf, char* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

    int comm_size, rank, mpi_errno = MPI_SUCCESS;
    int mask, dst, pof2, newrank, rem, newdst, i, send_idx, recv_idx, last_idx;

    int typesize;
    char* tmp_buf = NULL;

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);
    MPI_Type_size(datatype, &typesize);
    tmp_buf = (char*)malloc(count * typesize);

    if (sendbuf != MPI_IN_PLACE) {
        memcpy(recvbuf, sendbuf, count * typesize);
    }

    //get nearest power of 2 less or equal to comm_size
    pof2 = 1;
    while (pof2 <= comm_size) {
        pof2 *= 2;
    }
    pof2 /= 2;

    rem = comm_size - pof2;

    if (rank < 2 * rem){
        if(rank % 2 == 0){
            mpi_errno = MPI_Send(recvbuf, count, datatype, rank + 1, 0, comm);
            if (mpi_errno!= MPI_SUCCESS) {
                free(tmp_buf);
                return mpi_errno;
            }
            newrank = -1;;
        } else {
            mpi_errno = MPI_Recv(tmp_buf, count, datatype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
            if (mpi_errno!= MPI_SUCCESS) {
                free(tmp_buf);
                return mpi_errno;
            }
            mpi_errno = MPI_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
            if (mpi_errno!= MPI_SUCCESS) {
                free(tmp_buf);
                return mpi_errno;
            }
            newrank = rank /2;
        }
    }else{
        newrank = rank - rem;
    }

    if(newrank != -1){
        int *cnts, *disps;
        cnts = (int*)malloc(pof2 * sizeof(int));
        disps = (int*)malloc(pof2 * sizeof(int));

        for(i = 0; i < pof2; i++){
            cnts[i] = count/pof2;
        }
        if((count % pof2) > 0){
            for(i = 0; i < (count % pof2); i++){
                cnts[i]+=1;
            }
        }

        if(pof2)
            disps[0] = 0;
        for(i = 1; i < pof2; i++){
            disps[i] = disps[i-1] + cnts[i-1];
        }

        mask = 0x1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            MPI_Aint send_cnt, recv_cnt;
            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }

            mpi_errno = MPI_Sendrecv(recvbuf + disps[send_idx] * typesize, send_cnt, datatype, dst, 0, tmp_buf + disps[recv_idx] * typesize, recv_cnt, datatype, dst, 0, comm, MPI_STATUS_IGNORE);
            if (mpi_errno!= MPI_SUCCESS) {
                free(tmp_buf);
                return mpi_errno;
            }

            mpi_errno = MPI_Reduce_local(tmp_buf + disps[recv_idx] * typesize, recvbuf + disps[recv_idx] * typesize, recv_cnt, datatype, op);

            if (mpi_errno!= MPI_SUCCESS) {
                free(tmp_buf);
                return mpi_errno;
            }

            send_idx = recv_idx; // update send_idx for next roun
            mask <<= 1;

            if(mask < pof2){
                last_idx = recv_idx + pof2/mask;
            }

        }

        mask >>=1;
        while(mask > 0){
            newdst = newrank ^ mask;

            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;
            int send_cnt, recv_cnt;

            send_cnt = recv_cnt = 0;

            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pof2 / 2)
                    last_idx = last_idx + pof2 / (mask * 2);

                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }

            mpi_errno = MPI_Sendrecv(recvbuf + disps[send_idx] * typesize, send_cnt, datatype, dst, 0, recvbuf + disps[recv_idx] * typesize, recv_cnt, datatype, dst, 0, comm, MPI_STATUS_IGNORE);
            if (mpi_errno!= MPI_SUCCESS) {
                free(tmp_buf);
                return mpi_errno;
            }
            if(newrank > newdst){
                send_idx = recv_idx;
            }
            mask >>= 1;
        }
    }
    if(rank < 2 * rem){
        if(rank % 2){
            mpi_errno = MPI_Send(recvbuf, count, datatype, rank - 1, 0, comm);
        }else{
            mpi_errno = MPI_Recv(recvbuf, count, datatype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
        }
        if(mpi_errno!= MPI_SUCCESS){
            free(tmp_buf);
            return mpi_errno;
        }
    }
    free(tmp_buf);
    return MPI_SUCCESS;



}

#ifdef DEBUG_MODE
int main(int argc, char** argv) {
    int rank, size;
    int count = 100; // Default number of elements to reduce
    double *sendbuf, *recvbuf, *expected;
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
    
    if (rank == 0) {
        printf("Testing Reduce-Scatter-Allgather Allreduce with:\n");
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
    int rc = MPICH_Allreduce_reduce_scatter_allgather(
        (char*)sendbuf, (char*)recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    // Get maximum time across all processes
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rc != MPI_SUCCESS) {
        if (rank == 0) {
            printf("Error: MPICH_Allreduce_reduce_scatter_allgather returned %d\n", rc);
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
            printf("Custom implementation: PASSED, Time: %.9f seconds\n", max_time);
        } else {
            printf("Custom implementation: FAILED with %d errors, Time: %.9f seconds\n", 
                   total_errors, max_time);
        }
    }
    
    // Compare with standard MPI_Allreduce
    memset(recvbuf, 0, count * sizeof(double));
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    // Get maximum time across all processes
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Standard MPI_Allreduce: Time: %.9f seconds\n", max_time);
    }
    
    // Free memory
    free(sendbuf);
    free(recvbuf);
    free(expected);
    
    MPI_Finalize();
    return errors > 0 ? 1 : 0;
}
#endif




