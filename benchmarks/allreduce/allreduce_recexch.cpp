#include "common.hpp"






static int MPICH_Recexchalgo_get_neighbors(int rank, int nranks, int *k_, int *step1_sendto, int **step1_recvfrom_, int *step1_nrecvs, int ***step2_nbrs_, int *step2_nphases, int *p_of_k_, int *T_){
    int mpi_errno = MPI_SUCCESS;
    int i, j, k;
    int p_of_k = 1, log_p_of_k = 0, rem, T, newrank;
    int **step2_nbrs;
    int *step1_recvfrom;

    k = *k_;
    if (nranks < k)     /* If size of the communicator is less than k, reduce the value of k */
        k = (nranks > 2) ? nranks : 2;
    *k_ = k;

    while (p_of_k <= nranks) {
        p_of_k *= k;
        log_p_of_k++;
    }
    p_of_k /= k;
    log_p_of_k--;

    step1_recvfrom = *step1_recvfrom_ = (int *) malloc(sizeof(int) * (k - 1));
    step2_nbrs = *step2_nbrs_ = (int **) malloc(sizeof(int *) * log_p_of_k);



    if(step1_recvfrom == NULL || step2_nbrs == NULL) {
        std::cout<<"Error in malloc"<<std::endl;
        return MPI_ERR_OTHER;
    }

    for (i = 0; i < log_p_of_k; i++) {
        (*step2_nbrs_)[i] = (int *) malloc(sizeof(int) * (k - 1));
    }


    *step2_nphases = log_p_of_k;

    rem = nranks - p_of_k;

    T = (rem * k) / (k - 1);
    *T_ = T;
    *p_of_k_ = p_of_k;

    *step1_nrecvs = 0;
    *step1_sendto = -1;

    if (rank < T) {
        if (rank % k != (k - 1)) {      /* I am a non-participating rank */
            *step1_sendto = rank + (k - 1 - rank % k);  /* partipating rank to send the data to */
            /* if the corresponding participating rank is not in T,
             * then send to the Tth rank to preserve non-commutativity */
            if (*step1_sendto > T - 1)
                *step1_sendto = T;
            newrank = -1;       /* tag this rank as non-participating */
        } else {        /* participating rank */
            for (i = 0; i < k - 1; i++) {
                step1_recvfrom[i] = rank - i - 1;
            }
            *step1_nrecvs = k - 1;
            newrank = rank / k; /* this is the new rank amongst the set of participating ranks */
        }
    } else {    /* rank >= T */
        newrank = rank - rem;

        if (rank == T && (T - 1) % k != k - 1 && T >= 1) {
            int nsenders = (T - 1) % k + 1;     /* number of ranks sending their data to me in Step 1 */

            for (j = nsenders - 1; j >= 0; j--) {
                step1_recvfrom[nsenders - 1 - j] = T - nsenders + j;
            }
            *step1_nrecvs = nsenders;
        }
    }

    if (*step1_sendto == -1) {  /* calculate step2_nbrs only for participating ranks */
        int *digit = (int *) malloc(sizeof(int) * log_p_of_k);
        if(digit == NULL) {
            std::cout<<"Error in malloc"<<std::endl;
            return MPI_ERR_OTHER;
        }
        int temprank = newrank;
        int mask = 0x1;
        int phase = 0, cbit, cnt, nbr, power;

        /* calculate the digits in base k representation of newrank */
        for (i = 0; i < log_p_of_k; i++)
            digit[i] = 0;

        int remainder, i_digit = 0;
        while (temprank != 0) {
            remainder = temprank % k;
            temprank = temprank / k;
            digit[i_digit] = remainder;
            i_digit++;
        }

        while (mask < p_of_k) {
            cbit = digit[phase];        /* phase_th digit changes in this phase, obtain its original value */
            cnt = 0;
            for (i = 0; i < k; i++) {   /* there are k-1 neighbors */
                if (i != cbit) {        /* do not generate yourself as your nieighbor */
                    digit[phase] = i;   /* this gets us the base k representation of the neighbor */

                    /* calculate the base 10 value of the neighbor rank */
                    nbr = 0;
                    power = 1;
                    for (j = 0; j < log_p_of_k; j++) {
                        nbr += digit[j] * power;
                        power *= k;
                    }

                    /* calculate its real rank and store it */
                    step2_nbrs[phase][cnt] =
                        (nbr < rem / (k - 1)) ? (nbr * k) + (k - 1) : nbr + rem;
                    cnt++;
                }
            }

            digit[phase] = cbit;        /* reset the digit to original value */
            phase++;
            mask *= k;
        }

        free(digit);
    }

    return mpi_errno;

}

static int MPICH_find_myidx(int *nbrs, int k, int rank)
{
    for (int i = 0; i < k - 1; i++) {
        if (nbrs[i] > rank) {
            return i;
        }
    }
    return k - 1;
}

static int MPICH_do_reduce(void **bufs, void *recvbuf, int k, int idx,
                     MPI_Aint count, MPI_Datatype datatype, MPI_Op op)
{
    int mpi_errno = MPI_SUCCESS;

    for (int i = 0; i < idx - 1; i++) {
        mpi_errno = MPI_Reduce_local(bufs[i], bufs[i + 1], count, datatype, op);
        if (mpi_errno != MPI_SUCCESS) {
            return mpi_errno;
        }
    }
    if (idx > 0) {
        mpi_errno = MPI_Reduce_local(bufs[idx - 1], recvbuf, count, datatype, op);
        if (mpi_errno != MPI_SUCCESS) {
            return mpi_errno;
        }
    }
    if (idx < k - 1) {
        mpi_errno = MPI_Reduce_local(recvbuf, bufs[idx], count, datatype, op);
        if (mpi_errno != MPI_SUCCESS) {
            return mpi_errno;
        }

        for (int i = idx; i < k - 2; i++) {
            mpi_errno = MPI_Reduce_local(bufs[i], bufs[i + 1], count, datatype, op);
            if (mpi_errno != MPI_SUCCESS) {
                return mpi_errno;
            }
        }

        //mpi_errno = MPIR_Localcopy(bufs[k - 2], count, datatype, recvbuf, count, datatype);
        int extent;
        MPI_Type_size(datatype, &extent);
        memcpy(recvbuf, bufs[k - 2], count * extent);

    }

    return mpi_errno;

}

int MPICH_Allreduce_recursive_exchange(const char* sendbuf, char* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, int k, int single_phase_recv) {

    int mpi_errno = MPI_SUCCESS;
    int is_commutative, rank, nranks, nbr, myidx;
    int buf = 0;
    int extent;

    int step1_sendto = -1, step1_nrecvs = 0, *step1_recvfrom = NULL;
    int step2_nphases = 0, **step2_nbrs;
    int i, j, phase, p_of_k, T;
    bool in_step2 = true;
    void **nbr_buffer = NULL;
    int comm_alloc = 1;
    MPI_Request *send_reqs = NULL, *recv_reqs = NULL;
    int send_nreq = 0, recv_nreq = 0, total_phases = 0;

    MPI_Op_commutative(op, &is_commutative);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    MPI_Type_size(datatype, &extent);

    bool is_float;

    if (datatype == MPI_FLOAT || datatype == MPI_DOUBLE) {
        is_float = true;
    }

    if(nranks == 1) {
        if(sendbuf != MPI_IN_PLACE) {
            memcpy(recvbuf, sendbuf, count * extent);
        }
        return MPI_SUCCESS;
    }

    if(sendbuf!= MPI_IN_PLACE && count > 0) {
        memcpy(recvbuf, sendbuf, count * extent);
    }

    comm_alloc = 0;

    MPICH_Recexchalgo_get_neighbors(rank, nranks, &k, &step1_sendto, &step1_recvfrom, &step1_nrecvs, &step2_nbrs, &step2_nphases, &p_of_k, &T);

    if(step1_sendto == -1) {
        recv_reqs = (MPI_Request *) malloc(sizeof(MPI_Request) * 2 * (k-1));
        if(recv_reqs == NULL) {
            std::cout<<"Error in malloc"<<std::endl;
            return MPI_ERR_OTHER;
        }
        send_reqs = (MPI_Request *) malloc(sizeof(MPI_Request) * 2 * (k-1));
        if(send_reqs == NULL) {
            std::cout<<"Error in malloc"<<std::endl;
            return MPI_ERR_OTHER;
        }
    }

    in_step2 = (step1_sendto == -1);

    if(in_step2){
        if(single_phase_recv) {
            nbr_buffer = (void **) malloc(sizeof(void *) * (k - 1));
            if(nbr_buffer == NULL) {
                std::cout<<"Error in malloc"<<std::endl;
                return MPI_ERR_OTHER;
            }
            for(j = 0; j < k - 1; j++) {
                nbr_buffer[j] = malloc(count * extent);
                if(nbr_buffer[j] == NULL) {
                    std::cout<<"Error in malloc"<<std::endl;
                    return MPI_ERR_OTHER;
                }

            }
        }else {
            nbr_buffer = (void **) malloc(sizeof(void *) * 2 * (k - 1));
            if(nbr_buffer == NULL) {
                std::cout<<"Error in malloc"<<std::endl;
                return MPI_ERR_OTHER;
            }
            for(j = 0; j < 2 * (k - 1); j++) {
                nbr_buffer[j] = malloc(count * extent);
                if(nbr_buffer[j] == NULL) {
                    std::cout<<"Error in malloc"<<std::endl;
                    return MPI_ERR_OTHER;
                }
            }
        }
    }

    if(!in_step2){
        mpi_errno = MPI_Send(recvbuf, count, datatype, step1_sendto, 0, comm);
        if(mpi_errno != MPI_SUCCESS) {
            std::cout<<"Error in MPI_Send"<<std::endl;
            return mpi_errno;
        }
    }else{
        if(step1_nrecvs){
            for(i = 0; i < step1_nrecvs; i++){
                mpi_errno = MPI_Irecv(nbr_buffer[i], count, datatype, step1_recvfrom[i], 0, comm, &recv_reqs[recv_nreq++]);
                if(mpi_errno!= MPI_SUCCESS) {
                    std::cout<<"Error in MPI_Irecv"<<std::endl;
                    return mpi_errno;
                }
            }
            mpi_errno = MPI_Waitall(recv_nreq, recv_reqs, MPI_STATUSES_IGNORE);
            if(mpi_errno!= MPI_SUCCESS) {
                std::cout<<"Error in MPI_Waitall"<<std::endl;
                return mpi_errno;
            }
            for(i=0; i < step1_nrecvs && count > 0; i++){
                mpi_errno = MPI_Reduce_local(nbr_buffer[i], recvbuf, count, datatype, op);
                if(mpi_errno!= MPI_SUCCESS) {
                    std::cout<<"Error in MPI_Reduce_local"<<std::endl;
                    return mpi_errno;
                }
            }
        }
    }

    for(phase = 0; phase < step2_nphases && in_step2; phase++){
        buf = 0;
        recv_nreq = 0;
        total_phases = (single_phase_recv) ? 1 : 2;
        for (j = 0; j < total_phases && (j + phase) < step2_nphases; j++) {
            for (i = 0; i < (k - 1); i++) {
                nbr = step2_nbrs[phase + j][i];
                mpi_errno = MPI_Irecv(nbr_buffer[buf++], count, datatype, nbr, 0, comm, &recv_reqs[recv_nreq++]);
                if(mpi_errno!= MPI_SUCCESS) {
                    std::cout<<"Error in MPI_Irecv"<<std::endl;
                    return mpi_errno;
                }
            }
        }
        send_nreq = 0;
        for(i = 0; i < (k - 1); i++){
            nbr = step2_nbrs[phase][i];
            mpi_errno = MPI_Isend(recvbuf, count, datatype, nbr, 0, comm, &send_reqs[send_nreq++]);
            if(mpi_errno!= MPI_SUCCESS) {
                std::cout<<"Error in MPI_Isend"<<std::endl;
                return mpi_errno;
            }

        }

         mpi_errno = MPI_Waitall(send_nreq, send_reqs, MPI_STATUSES_IGNORE);
        if(mpi_errno!= MPI_SUCCESS) {
            std::cout<<"Error in MPI_Waitall"<<std::endl;
            return mpi_errno;
        }
        mpi_errno = MPI_Waitall(k -1, recv_reqs, MPI_STATUSES_IGNORE);
        if(mpi_errno!= MPI_SUCCESS) {
            std::cout<<"Error in MPI_Waitall"<<std::endl;
            return mpi_errno;
        }

        if (is_commutative && !is_float) {
            myidx = k - 1;
        } else {
            myidx = MPICH_find_myidx(step2_nbrs[phase], k, rank);
        }
        mpi_errno = MPICH_do_reduce(nbr_buffer, recvbuf, k, myidx, count, datatype, op);

        if(mpi_errno!= MPI_SUCCESS) {
            std::cout<<"Error in MPICH_do_reduce"<<std::endl;
            return mpi_errno;
        }

        if(single_phase_recv == false){
            phase ++;
            if(phase < step2_nphases){
                send_nreq = 0;

                for(i = 0; i < k -1; i++){
                    nbr = step2_nbrs[phase][i];
                    mpi_errno = MPI_Isend(recvbuf, count, datatype, nbr, 0, comm, &send_reqs[send_nreq++]);
                    if(mpi_errno!= MPI_SUCCESS) {
                        std::cout<<"Error in MPI_Isend"<<std::endl;
                        return mpi_errno;
                    }
                }
                mpi_errno = MPI_Waitall(send_nreq, send_reqs, MPI_STATUSES_IGNORE);
                if(mpi_errno!= MPI_SUCCESS) {
                    std::cout<<"Error in MPI_Waitall"<<std::endl;
                    return mpi_errno;
                }
                mpi_errno = MPI_Waitall(k - 1, recv_reqs + (k-1), MPI_STATUSES_IGNORE);
                if(mpi_errno!= MPI_SUCCESS) {
                    std::cout<<"Error in MPI_Waitall"<<std::endl;
                    return mpi_errno;
                }
                if (is_commutative && !is_float) {
                    myidx = k - 1;
                } else {
                    myidx = MPICH_find_myidx(step2_nbrs[phase], k, rank);
                }
                mpi_errno = MPICH_do_reduce(nbr_buffer + k -1, recvbuf, k, myidx, count, datatype, op);
                if(mpi_errno!= MPI_SUCCESS) {
                    std::cout<<"Error in MPICH_do_reduce"<<std::endl;
                    return mpi_errno;
                }

            }
        }
    }

    if(step1_sendto != -1){
        mpi_errno = MPI_Recv(recvbuf, count, datatype, step1_sendto, 0, comm, MPI_STATUS_IGNORE);
        if(mpi_errno!= MPI_SUCCESS) {
            std::cout<<"Error in MPI_Recv"<<std::endl;
            return mpi_errno;
        }
    }else{
        for(i = 0; i < step1_nrecvs; i++){
            mpi_errno = MPI_Isend(recvbuf, count, datatype, step1_recvfrom[i], 0, comm, &send_reqs[i]);
            if(mpi_errno!= MPI_SUCCESS) {
                std::cout<<"Error in MPI_Isend"<<std::endl;
                return mpi_errno;
            }
        }
        mpi_errno = MPI_Waitall(i, send_reqs, MPI_STATUSES_IGNORE);
        if(mpi_errno!= MPI_SUCCESS) {
            std::cout<<"Error in MPI_Waitall"<<std::endl;
            return mpi_errno;
        }

    }

    if (in_step2) {
        if (single_phase_recv == true) {
            for (j = 0; j < (k - 1); j++) {
                nbr_buffer[j] = (void *) ((char *) nbr_buffer[j]);
                free(nbr_buffer[j]);
            }
        } else {
            for (j = 0; j < 2 * (k - 1); j++) {
                nbr_buffer[j] = (void *) ((char *) nbr_buffer[j]);
                free(nbr_buffer[j]);
            }
        }
        free(nbr_buffer);
    }

    if (!comm_alloc) {
        /* free the memory */
        for (i = 0; i < step2_nphases; i++)
            free(step2_nbrs[i]);
        free(step2_nbrs);
        free(step1_recvfrom);
        if (in_step2) {
            free(recv_reqs);
            free(send_reqs);
        }
    }


    return mpi_errno;




}

#ifdef DEBUG_MODE
int main(int argc, char** argv) {
    int rank, size;
    int count = 10; // Number of elements to reduce
    double *sendbuf, *recvbuf, *expected, *mpi_result;
    int i, j, errors = 0;
    int k = 3; // Default radix for recursive exchange
    int single_phase_recv = 0; // Default to two-phase receive
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Process command line arguments
    if (argc > 1) {
        count = atoi(argv[1]);
    }
    if (argc > 2) {
        k = atoi(argv[2]);
    }
    if (argc > 3) {
        single_phase_recv = atoi(argv[3]);
    }
    
    if (rank == 0) {
        printf("Testing Recursive Exchange Allreduce with count=%d, k=%d, single_phase_recv=%d\n", 
               count, k, single_phase_recv);
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
    
    // Call our custom recursive exchange allreduce implementation
    int result = MPICH_Allreduce_recursive_exchange((char*)sendbuf, (char*)recvbuf, 
                                                    count, MPI_DOUBLE, MPI_SUM, 
                                                    MPI_COMM_WORLD, k, single_phase_recv);
    
    if (result != MPI_SUCCESS) {
        if (rank == 0) {
            printf("ERROR: MPICH_Allreduce_recursive_exchange failed with error code %d\n", result);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
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
    
    // Test with different radix values
    int test_k_values[] = {2, 4, 8};
    int num_k_tests = sizeof(test_k_values) / sizeof(test_k_values[0]);
    
    for (int test = 0; test < num_k_tests; test++) {
        int test_k = test_k_values[test];
        
        if (rank == 0) {
            printf("\nTesting with k=%d...\n", test_k);
        }
        
        // Reset buffers
        for (i = 0; i < count; i++) {
            sendbuf[i] = rank + i * 0.1;
            recvbuf[i] = 0.0;
        }
        
        result = MPICH_Allreduce_recursive_exchange((char*)sendbuf, (char*)recvbuf, 
                                                    count, MPI_DOUBLE, MPI_SUM, 
                                                    MPI_COMM_WORLD, test_k, single_phase_recv);
        
        if (result != MPI_SUCCESS) {
            if (rank == 0) {
                printf("ERROR: Test with k=%d failed with error code %d\n", test_k, result);
            }
            continue;
        }
        
        // Verify results
        errors = 0;
        for (i = 0; i < count; i++) {
            if (fabs(recvbuf[i] - expected[i]) > 1e-10) {
                errors++;
            }
        }
        
        if (rank == 0) {
            if (errors == 0) {
                printf("Test with k=%d PASSED\n", test_k);
            } else {
                printf("Test with k=%d FAILED: %d errors\n", test_k, errors);
            }
        }
    }
    
    // Test both single-phase and two-phase receive modes
    for (int phase_mode = 0; phase_mode <= 1; phase_mode++) {
        if (rank == 0) {
            printf("\nTesting %s receive mode...\n", 
                   phase_mode ? "single-phase" : "two-phase");
        }
        
        // Reset buffers
        for (i = 0; i < count; i++) {
            sendbuf[i] = rank + i * 0.1;
            recvbuf[i] = 0.0;
        }
        
        result = MPICH_Allreduce_recursive_exchange((char*)sendbuf, (char*)recvbuf, 
                                                    count, MPI_DOUBLE, MPI_SUM, 
                                                    MPI_COMM_WORLD, k, phase_mode);
        
        if (result != MPI_SUCCESS) {
            if (rank == 0) {
                printf("ERROR: %s mode failed with error code %d\n", 
                       phase_mode ? "Single-phase" : "Two-phase", result);
            }
            continue;
        }
        
        // Verify results
        errors = 0;
        for (i = 0; i < count; i++) {
            if (fabs(recvbuf[i] - expected[i]) > 1e-10) {
                errors++;
            }
        }
        
        if (rank == 0) {
            if (errors == 0) {
                printf("%s mode PASSED\n", phase_mode ? "Single-phase" : "Two-phase");
            } else {
                printf("%s mode FAILED: %d errors\n", 
                       phase_mode ? "Single-phase" : "Two-phase", errors);
            }
        }
    }
    
    // Clean up
    free(sendbuf);
    free(recvbuf);
    free(expected);
    free(mpi_result);
    
    if (rank == 0) {
        printf("\nRecursive Exchange Allreduce testing completed.\n");
    }
    
    MPI_Finalize();
    return 0;
}
#endif

