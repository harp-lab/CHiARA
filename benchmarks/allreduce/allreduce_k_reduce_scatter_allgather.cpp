#include "common.hpp"





static int MPICH_Recexchalgo_step2rank_to_origrank(int rank, int rem, int T, int k)
{
    int orig_rank;

    orig_rank = (rank < rem / (k - 1)) ? (rank * k) + (k - 1) : rank + rem;

    return orig_rank;
}

static int MPICH_Recexchalgo_origrank_to_step2rank(int rank, int rem, int T, int k)
{
    int step2rank;

    step2rank = (rank < T) ? rank / k : rank - rem;

    return step2rank;
}



static int MPICH_Recexchalgo_get_count_and_offset(int rank, int phase, int k, int nranks, int *count,
                                          int *offset)
{
    int mpi_errno = MPI_SUCCESS;
    int step2rank, min, max, orig_max, orig_min;
    int k_power_phase = 1;
    int p_of_k = 1, rem, T;


    /* p_of_k is the largest power of k that is less than nranks */
    while (p_of_k <= nranks) {
        p_of_k *= k;
    }
    p_of_k /= k;

    rem = nranks - p_of_k;
    T = (rem * k) / (k - 1);

    /* k_power_phase is k^phase */
    while (phase > 0) {
        k_power_phase *= k;
        phase--;
    }
    /* Calculate rank in step2 */
    step2rank = MPICH_Recexchalgo_origrank_to_step2rank(rank, rem, T, k);
    /* Calculate min and max ranks of the range of ranks that 'rank'
     * represents in phase 'phase' */
    min = ((step2rank / k_power_phase) * k_power_phase) - 1;
    max = min + k_power_phase;
    /* convert (min,max] to their original ranks */
    orig_min = (min >= 0) ? MPICH_Recexchalgo_step2rank_to_origrank(min, rem, T, k) : min;
    orig_max = MPICH_Recexchalgo_step2rank_to_origrank(max, rem, T, k);
    *count = orig_max - orig_min;
    *offset = orig_min + 1;



    return mpi_errno;
}

static int MPICH_Recexchalgo_reverse_digits_step2(int rank, int comm_size, int k)
{
    int i, T, rem, power, step2rank, step2_reverse_rank = 0;
    int pofk = 1, log_pofk = 0;
    int *digit, *digit_reverse;


    while (pofk <= comm_size) {
        pofk *= k;
        log_pofk++;
    }
    if(!(log_pofk > 0)) {
        std::cout<<"Error in calculating log_pofk"<<std::endl;
        return MPI_ERR_OTHER;
    }
    pofk /= k;
    log_pofk--;

    rem = comm_size - pofk;
    T = (rem * k) / (k - 1);

    /* step2rank is the rank in the particiapting ranks group of recursive exchange step2 */
    step2rank = MPICH_Recexchalgo_origrank_to_step2rank(rank, rem, T, k);

    /* calculate the digits in base k representation of step2rank */
    digit = (int*) malloc(sizeof(int) * log_pofk);
    digit_reverse = (int*) malloc( sizeof(int) * log_pofk);

    for (i = 0; i < log_pofk; i++)
        digit[i] = 0;

    int remainder, i_digit = 0;
    while (step2rank != 0) {
        remainder = step2rank % k;
        step2rank = step2rank / k;
        digit[i_digit] = remainder;
        i_digit++;
    }

    /* reverse the number in base k representation to get the step2_reverse_rank
     * which is the reversed rank in the participating ranks group of recursive exchange step2
     */
    for (i = 0; i < log_pofk; i++)
        digit_reverse[i] = digit[log_pofk - 1 - i];
    /*calculate the base 10 value of the reverse rank */
    step2_reverse_rank = 0;
    power = 1;
    for (i = 0; i < log_pofk; i++) {
        step2_reverse_rank += digit_reverse[i] * power;
        power *= k;
    }

    /* calculate the actual rank from logical rank */
    step2_reverse_rank = MPICH_Recexchalgo_step2rank_to_origrank(step2_reverse_rank, rem, T, k);

    free(digit);
    free(digit_reverse);
    return step2_reverse_rank;
}

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

int MPICH_Allreduce_k_reduce_scatter_allgather(const char* sendbuf, char* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,int k, int single_phase_recv) {

    int mpi_errno = MPI_SUCCESS;
    int rank, nranks, nbr;
    int rem = 0, idx = 0, dst = 0, rank_for_offset;
    int extent;
    int current_cnt = 0, send_count = 0, recv_count = 0, send_cnt = 0, recv_cnt = 0, iter = 0;
    int offset = 0;
    int step1_sendto = -1, step1_nrecvs = 0, *step1_recvfrom = NULL;
    int step2_nphases = 0, **step2_nbrs;
    int i, j, x, phase, recv_phase, p_of_k, T;
    bool in_step2 = true;
    int comm_alloc = 1;
    MPI_Request *send_reqs = NULL, *recv_reqs = NULL;
    int num_sreq = 0, num_rreq = 0, total_phases = 0;
    char *tmp_recvbuf = NULL;

    if(k <= 1){
        k = 2;
    }

    //check if op is commutative
    MPI_Op_commutative(op, &mpi_errno);
    if(mpi_errno != 1){
        std::cout<<"Op is not commutative"<<std::endl;
        return MPI_ERR_OP;
    }

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    MPI_Type_size(datatype, &extent);

    if(sendbuf != MPI_IN_PLACE && count > 0) {
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

    tmp_recvbuf = (char *) malloc(count * extent);
    if(tmp_recvbuf == NULL) {
        std::cout<<"Error in malloc"<<std::endl;
        return MPI_ERR_OTHER;
    }

    in_step2 = (step1_sendto == -1);
    if(!in_step2) {
        mpi_errno = MPI_Send(recvbuf, count, datatype, step1_sendto, 0, comm);
        if(mpi_errno!= MPI_SUCCESS) {
            std::cout<<"Error in MPI_Send"<<std::endl;
            return mpi_errno;
        }
    }else{
        for(i = 0; i < step1_nrecvs; i++) {
            mpi_errno = MPI_Recv(tmp_recvbuf, count, datatype, step1_recvfrom[i], 0, comm, MPI_STATUS_IGNORE);
            if(mpi_errno!= MPI_SUCCESS) {
                std::cout<<"Error in MPI_Recv"<<std::endl;
                return mpi_errno;
            }
            mpi_errno = MPI_Reduce_local(tmp_recvbuf, recvbuf, count, datatype, op);
            if(mpi_errno!= MPI_SUCCESS) {
                std::cout<<"Error in MPI_Reduce_local"<<std::endl;
                return mpi_errno;
            }
        }
    }

    if (in_step2){
        int *cnts, *displs;
        cnts = (int *) malloc(sizeof(int) * nranks);
        displs = (int *) malloc(sizeof(int) * nranks);
        idx = 0;
        rem = nranks - p_of_k;

        for (i = 0; i < nranks; i++)
        cnts[i] = 0;
        for (i = 0; i < (p_of_k - 1); i++) {
            idx = (i < rem / (k - 1)) ? (i * k) + (k - 1) : i + rem;
            cnts[idx] = count / p_of_k;
        }
        idx = (p_of_k - 1 < rem / (k - 1)) ? (p_of_k - 1 * k) + (k - 1) : p_of_k - 1 + rem;
        cnts[idx] = count - (count / p_of_k) * (p_of_k - 1);

        displs[0] = 0;
        for (i = 1; i < nranks; i++) {
            displs[i] = displs[i - 1] + cnts[i - 1];
        }

        for (phase = 0, j = step2_nphases - 1; phase < step2_nphases; j--, phase++) {
            /* send data to all the neighbors */
            for (i = 0; i < k - 1; i++) {
                num_rreq = 0;
                dst = step2_nbrs[phase][i];

                rank_for_offset = MPICH_Recexchalgo_reverse_digits_step2(dst, nranks, k);
                MPICH_Recexchalgo_get_count_and_offset(rank_for_offset, j, k, nranks,
                                                      &current_cnt, &offset);
                MPI_Aint send_offset = displs[offset] * extent;
                send_cnt = 0;
                for (x = 0; x < current_cnt; x++)
                    send_cnt += cnts[offset + x];

                mpi_errno = MPI_Isend(recvbuf + send_offset, send_cnt, datatype, dst, 0, comm, &recv_reqs[num_rreq++]);
                if(mpi_errno!= MPI_SUCCESS) {
                    std::cout<<"Error in MPI_Isend"<<std::endl;
                    return mpi_errno;
                }
                rank_for_offset = MPICH_Recexchalgo_reverse_digits_step2(rank, nranks, k);
                MPICH_Recexchalgo_get_count_and_offset(rank_for_offset, j, k, nranks,
                                                      &current_cnt, &offset);

                MPI_Aint recv_offset = displs[offset] * extent;
                recv_cnt = 0;
                for (x = 0; x < current_cnt; x++)
                    recv_cnt += cnts[offset + x];


                mpi_errno = MPI_Irecv(tmp_recvbuf + recv_offset, recv_cnt, datatype, dst, 0, comm, &recv_reqs[num_rreq++]);
                if(mpi_errno!= MPI_SUCCESS) {
                    std::cout<<"Error in MPI_Irecv"<<std::endl;
                    return mpi_errno;
                }
                mpi_errno = MPI_Waitall(num_rreq, recv_reqs, MPI_STATUSES_IGNORE);
                if(mpi_errno!= MPI_SUCCESS) {
                    std::cout<<"Error in MPI_Waitall"<<std::endl;
                    return mpi_errno;
                }
                mpi_errno = MPI_Reduce_local(tmp_recvbuf + recv_offset, recvbuf + recv_offset, recv_cnt, datatype, op);

                if(mpi_errno!= MPI_SUCCESS) {
                    std::cout<<"Error in MPI_Reduce_local"<<std::endl;
                    return mpi_errno;
                }
            }
        }

        phase = step2_nphases - 1;
        recv_phase = step2_nphases - 1;
        for (j = 0; j < step2_nphases && step1_sendto == -1; j++) {
            /* post recvs for two phases. If optimization enabled, post for one phase  */
            total_phases = (single_phase_recv) ? 1 : 2;
            num_sreq = 0;
            num_rreq = 0;
            for (iter = 0; iter < total_phases && (j + iter) < step2_nphases; iter++) {
                for (i = 0; i < k - 1; i++) {
                    nbr = step2_nbrs[recv_phase][i];
                    rank_for_offset = MPICH_Recexchalgo_reverse_digits_step2(nbr, nranks, k);

                    MPICH_Recexchalgo_get_count_and_offset(rank_for_offset, j + iter, k, nranks,
                                                          &current_cnt, &offset);
                    MPI_Aint recv_offset = displs[offset] * extent;
                    recv_count = 0;
                    for (x = 0; x < current_cnt; x++)
                        recv_count += cnts[offset + x];
                    mpi_errno = MPI_Irecv(( recvbuf + recv_offset), recv_count, datatype,
                                           nbr, 0, comm, &recv_reqs[num_rreq++]);
                    if(mpi_errno!= MPI_SUCCESS) {
                        std::cout<<"Error in MPI_Irecv"<<std::endl;
                        return mpi_errno;
                    }
                }
                recv_phase--;
            }
            /* send data to all the neighbors */
            for (i = 0; i < k - 1; i++) {
                nbr = step2_nbrs[phase][i];
                rank_for_offset = MPICH_Recexchalgo_reverse_digits_step2(rank, nranks, k);
                MPICH_Recexchalgo_get_count_and_offset(rank_for_offset, j, k, nranks, &current_cnt,
                                                      &offset);
                MPI_Aint send_offset = displs[offset] * extent;
                send_count = 0;
                for (x = 0; x < current_cnt; x++)
                    send_count += cnts[offset + x];
                mpi_errno = MPI_Isend((recvbuf + send_offset), send_count, datatype,
                                       nbr, 0, comm, &send_reqs[num_sreq++]);
                if(mpi_errno!= MPI_SUCCESS) {
                    std::cout<<"Error in MPI_Isend"<<std::endl;
                    return mpi_errno;
                }
            }
            /* wait on prev recvs */
            mpi_errno = MPI_Waitall((k - 1), recv_reqs, MPI_STATUSES_IGNORE);
            if(mpi_errno!= MPI_SUCCESS) {
                std::cout<<"Error in MPI_Waitall"<<std::endl;
                return mpi_errno;
            }

            phase--;
            if (single_phase_recv == false) {
                j++;
                if (j < step2_nphases) {
                    /* send data to all the neighbors */
                    for (i = 0; i < k - 1; i++) {
                        nbr = step2_nbrs[phase][i];
                        rank_for_offset = MPICH_Recexchalgo_reverse_digits_step2(rank, nranks, k);
                        MPICH_Recexchalgo_get_count_and_offset(rank_for_offset, j, k, nranks,
                                                              &current_cnt, &offset);
                        MPI_Aint send_offset = displs[offset] * extent;
                        send_count = 0;
                        for (x = 0; x < current_cnt; x++)
                            send_count += cnts[offset + x];
                        mpi_errno =
                            MPI_Isend((recvbuf + send_offset), send_count, datatype, nbr,
                                       0, comm, &send_reqs[num_sreq++]);
                        if(mpi_errno!= MPI_SUCCESS) {
                            std::cout<<"Error in MPI_Isend"<<std::endl;
                            return mpi_errno;
                        }
                    }
                    /* wait on prev recvs */
                    mpi_errno = MPI_Waitall((k - 1), recv_reqs + (k - 1), MPI_STATUSES_IGNORE);
                    if(mpi_errno!= MPI_SUCCESS) {
                        std::cout<<"Error in MPI_Waitall"<<std::endl;
                        return mpi_errno;
                    }

                    phase--;
                }
            }
            mpi_errno = MPI_Waitall(num_sreq, send_reqs, MPI_STATUSES_IGNORE);
            if(mpi_errno!= MPI_SUCCESS) {
                std::cout<<"Error in MPI_Waitall"<<std::endl;
                return mpi_errno;
            }
        }
    }

    if (step1_sendto != -1) {   /* I am a Step 2 non-participating rank */
        mpi_errno = MPI_Recv(recvbuf, count, datatype, step1_sendto, 0, comm,
                              MPI_STATUS_IGNORE);
        if(mpi_errno!= MPI_SUCCESS) {
            std::cout<<"Error in MPI_Recv"<<std::endl;
            return mpi_errno;
        }
    } else {
        if (step1_nrecvs > 0) {
            for (i = 0; i < step1_nrecvs; i++) {
                mpi_errno =
                    MPI_Isend(recvbuf, count, datatype, step1_recvfrom[i], 0,
                               comm, &send_reqs[i]);
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
    free(tmp_recvbuf);

    return mpi_errno;

}

#ifdef DEBUG_MODE

int main(int argc, char** argv) {
    int rank, size;
    int count = 100; // Default number of elements to reduce
    int k = 3;      // Default radix value
    bool single_phase_recv = false; // Default to false for the new parameter
    double *sendbuf, *recvbuf, *expected;
    int i, errors = 0;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Process command line arguments for count, k, and single_phase_recv
    if (argc > 1) {
        count = atoi(argv[1]);
    }
    
    if (argc > 2) {
        k = atoi(argv[2]);
        if (k < 2) {
            if (rank == 0) {
                std::cout << "k must be at least 2. Setting k=2." << std::endl;
            }
            k = 2;
        }
    }
    
    if (argc > 3) {
        single_phase_recv = (atoi(argv[3]) != 0); // Any non-zero value enables single phase receive
    }
    
    // Allocate memory for buffers
    sendbuf = new double[count];
    recvbuf = new double[count];
    expected = new double[count];
    
    // Initialize send buffer with rank-specific values
    for (i = 0; i < count; i++) {
        sendbuf[i] = rank + i * 0.1;
    }
    
    // Calculate expected result (sum of all ranks' values)
    for (i = 0; i < count; i++) {
        expected[i] = 0;
        for (int r = 0; r < size; r++) {
            expected[i] += r + i * 0.1;
        }
    }
    
    // Print test information
    if (rank == 0) {
        std::cout << "Testing k-ary Reduce-Scatter-Allgather Allreduce with:" << std::endl;
        std::cout << "  Processes: " << size << std::endl;
        std::cout << "  Elements per process: " << count << std::endl;
        std::cout << "  Radix (k): " << k << std::endl;
        std::cout << "  Single phase receive: " << (single_phase_recv ? "enabled" : "disabled") << std::endl;
    }
    
    // Call the k-ary reduce-scatter-allgather allreduce implementation with the new parameter
    MPICH_Allreduce_k_reduce_scatter_allgather((char*)sendbuf, (char*)recvbuf, count, 
                                              MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, k, single_phase_recv);
    
    // Verify results
    for (i = 0; i < count; i++) {
        if (fabs(recvbuf[i] - expected[i]) > 1e-10) {
            errors++;
            if (errors < 10) { // Limit error output
                std::cout << "Rank " << rank << ": Error at index " << i 
                          << ", expected " << expected[i] 
                          << ", got " << recvbuf[i] << std::endl;
            }
        }
    }
    
    // Report results
    int total_errors;
    MPI_Reduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (total_errors == 0) {
            std::cout << "Test PASSED!" << std::endl;
        } else {
            std::cout << "Test FAILED with " << total_errors << " errors." << std::endl;
        }
    }
    
    // Clean up
    delete[] sendbuf;
    delete[] recvbuf;
    delete[] expected;
    
    MPI_Finalize();
    return 0;
}

#endif // DEBUG_MODE