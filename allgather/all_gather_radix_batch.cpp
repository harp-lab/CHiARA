#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cstring>   // for std::memcpy / memcpy

#define CHECK_ALLOCATIONS(...) \
    if ((__VA_ARGS__)) { \
        std::cerr << "Memory allocation failed!" << std::endl; \
        return 1; \
    }

#define CHECK_MPI_ERROR(...) \
    if ((__VA_ARGS__)) { \
        std::cerr << "MPI error " << std::endl; \
        return 1; \
    }


int ipow(int base, int exp) {
    int result = 1;
    while (exp) {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }
    return result;
}

int min(int a, int b) {
    return (a < b) ? a : b;
}


int allgather_radix_batch(char* sendbuf, int sendcount, MPI_Datatype datatype, char* recvbuf, MPI_Comm comm, int k, int b){
    int rank, nprocs, typeSize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(datatype, &typeSize);
    int num_reqs = 0;
    int count= sendcount * b;

    int node_id = rank/b;
    int node_rank = rank%b;
    int nnodes = nprocs/b;
    const int nstages = nnodes/b;
    int group = rank / b;

    int i,j;

    ////---------- PHASE 1 INTRA GATHER K-NOMIAL----------------
    int root_local = node_id%b;
    int shift = (node_rank - root_local + b) % b;
    int max = b - 1, nphases = 0;
    while (max > 0) { ++nphases; max /= k; }
    bool p_of_k = (ipow(k, nphases) == b);

    char* kgather_tmp_buf = (char*)malloc(b * sendcount * typeSize);
    if (!kgather_tmp_buf) return MPI_ERR_NO_MEM;

    char* inter_tmp_buf = (char*)malloc(b * sendcount * typeSize);
    if (!inter_tmp_buf) return MPI_ERR_NO_MEM;

    std::memcpy(kgather_tmp_buf + shift * sendcount * typeSize,
            sendbuf,
            sendcount * typeSize);

    // scratch array for up to (k-1) outstanding Irecvs
    MPI_Request* reqs = (MPI_Request*)malloc(((nstages+1)*(1+nnodes)+2*(k - 1)) * sizeof(MPI_Request));

    if (!reqs) { free(kgather_tmp_buf); return MPI_ERR_NO_MEM; }

    int phase,delta;
    int src;

    for (phase = 0, delta = 1; phase < nphases; ++phase, delta *= k) {
        int groupSize  = delta * k;
        int gstart     = (shift / groupSize) * groupSize;
        int offset     = shift - gstart;

        // real‐world rank of the normalized 0-root:
        int real_root_local = (gstart + root_local) % b;
        int real_root_glob  = node_id * b + real_root_local;

        if (offset == 0) {
            // I am this subtree’s root → post up to (k−1) Irecvs
            num_reqs =0;
            for (j = 1; j < k; ++j) {
                int childNorm = gstart + j*delta;
                if (childNorm >= b) break;
                int subtree = delta;
                if (/*phase == nphases - 1 && */!p_of_k)
                    subtree = std::min(subtree, b - childNorm);

                // recv into the *normalized* slot “childNorm”
                int src_local = (childNorm + root_local) % b;
                src  = node_id * b + src_local;
                MPI_Irecv(kgather_tmp_buf + childNorm * sendcount * typeSize,
                          subtree * sendcount, datatype,
                          src, phase,
                          comm, &reqs[num_reqs++]);
            }
            MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
        }
        else if ((offset % delta) == 0 && offset < groupSize) {
            // I am a child‐subtree root → send my normalized block up
            int subtree = delta;
            if (/*phase == nphases - 1 && */!p_of_k)
                subtree = std::min(subtree, b - shift);

            MPI_Request sreq;
            MPI_Isend(kgather_tmp_buf + shift * sendcount * typeSize,
                      subtree * sendcount, datatype,
                      real_root_glob, phase,
                      comm, &sreq);
            MPI_Wait(&sreq, MPI_STATUS_IGNORE);
            break;
        }
    }

    if (node_rank == root_local) {
        for (int real_slot = 0; real_slot < b; ++real_slot) {
            // find which normalized index holds that real_slot
            int norm_i = (real_slot - root_local + b) % b;
            std::memcpy(inter_tmp_buf + real_slot * sendcount * typeSize,
                        kgather_tmp_buf + norm_i    * sendcount * typeSize,
                        sendcount * typeSize);
        }
    }
    


    ////-------------------------------------------------------

    

    /// ------------- PHASE 2 INTER ALLGATHER LINEAR-----------

    bool rootCond;
    

    char* intra_tmp_buf = (char*)malloc((nstages + 1)*count*typeSize);
    if (!intra_tmp_buf) return MPI_ERR_NO_MEM;


    for(i = 0; i<nnodes; i+=b){
        num_reqs = 0;
        rootCond = ((node_id - i) == node_rank);
        if(!rootCond && (i+node_rank < nnodes)){
            MPI_Irecv(intra_tmp_buf+count*(i/b)*typeSize, count, datatype, (i+node_rank)*b + node_rank, 1, comm, &reqs[num_reqs++]);
        }else if(rootCond){
            memcpy(intra_tmp_buf+count*(i/b)*typeSize, inter_tmp_buf, count * typeSize);
                for(j=0; j<nnodes; j++){
                    if(j==node_id )
                        continue;
                    int dst = j*b + node_rank;
                    MPI_Isend(inter_tmp_buf, count, datatype, dst, 1, comm, &reqs[num_reqs++]);
                }
        }
        MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
    }




    ///--------------------------------------------------------
    

    ///-----------PHASE 3 INTRA ALLGATHER K BRUCKS-------------

    char* tmp_recv_buffer;
    

    if(node_rank == 0) {
        tmp_recv_buffer = recvbuf;
    } else {
        tmp_recv_buffer = (char*) malloc((nstages*b + nnodes%b) * count * typeSize);  // Fixed 'size' to 'nprocs'
        if (tmp_recv_buffer == NULL) {
            free(reqs);
            return MPI_ERR_NO_MEM;
        }
    }

    int s,dst;
    delta = 1;

    for(s = 0; s < nstages; s++){

        memcpy(tmp_recv_buffer, intra_tmp_buf, count * typeSize);
        
        for(i = 0; i < nphases; i++) {
            num_reqs = 0;
            for(j = 1; j < k; j++) {
                if(delta * j >= b)  // Fixed 'size' to 'nprocs'
                    break;
                    
                dst = (b + (node_rank - delta * j)) % b + group*b;
                src = (node_rank + delta * j) % b + group*b;  // Fixed 'size' to 'nprocs' and 'k' to 'j'

                int tmp_count;
                if ((i == nphases - 1) && (!p_of_k)) {
                    tmp_count = count * delta;
                    int left_count = count * (b - delta * j);
                    if (j == k - 1) {
                        tmp_count = left_count;
                    } else {
                        tmp_count = min(tmp_count, left_count);
                    }
                } else {
                    tmp_count = count * delta;
                }

                MPI_Irecv(tmp_recv_buffer + j * count * delta * typeSize, 
                            tmp_count, datatype, src, 1, comm, &reqs[num_reqs++]);

                MPI_Isend(tmp_recv_buffer, tmp_count, datatype, 
                            dst, 1, comm, &reqs[num_reqs++]);
            }
            
            MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
            delta *= k;
        }

        if(node_rank != 0) {
            // Rearrange data in the correct order
            memcpy(recvbuf, 
                    tmp_recv_buffer + (b - node_rank) * count * typeSize, 
                    node_rank * count * typeSize);

            memcpy(recvbuf + node_rank * count * typeSize, 
                    tmp_recv_buffer, 
                    (b - node_rank) * count * typeSize);
                    
            //free(tmp_recv_buffer);
        }
        delta = 1;

        tmp_recv_buffer += b * count * typeSize;
        recvbuf += b*count*typeSize;
        intra_tmp_buf += count*typeSize;
    }

    int nu_count = nnodes%b;

    if(nu_count != 0) {

        if(node_rank < nu_count) {
            memcpy(tmp_recv_buffer, intra_tmp_buf, count * typeSize);
        } else {
            free(tmp_recv_buffer-nstages*b*count*typeSize);
            tmp_recv_buffer = recvbuf;
        }

        int active[b];
        int send_sizes[nphases+1][b];
        memset(send_sizes, 0, (nphases+1)*b*sizeof(int));
        //sizes array

        for(i =0; i< b; i++){
            active[i] = i < nu_count ? 0 : -1;
            send_sizes[0][i] = i < nu_count ? count : 0;

            //initialize sizes to 1 block if i < nnodes%b  nproc/b 0 otherwise 
        }



        int isrc,idst;
        int received;

        for(i = 0; i < nphases; i++) {
            num_reqs = 0;

            received = send_sizes[i][node_rank];
            
            for(j = 1; j < k; j++) {
                if(delta * j >= b){

                    for(int l = 0; l < b; l++){
                        if(active[l] == i){
                                active[l] = i+1;

                        }
                        send_sizes[i+1][l] += send_sizes[i][l];
                    }


                    break;

                }  

                

                

                isrc = (node_rank + delta * j) % b;
                idst = (b + (node_rank - delta * j)) % b;
                    
                dst = idst + group*b;
                src = isrc + group*b;  // Fixed 'size' to 'nprocs' and 'k' to 'j'

                
                int ssssize;
                ssssize = min(send_sizes[i][isrc], (nu_count) * count - received);
                if(active[isrc] == i && ssssize > 0){
                    MPI_Irecv(tmp_recv_buffer + received * typeSize, 
                            ssssize, datatype, src, 1, comm, &reqs[num_reqs++]);
                    
                    received += ssssize;
                }
                ssssize = min(send_sizes[i][node_rank], (nu_count) * count - (send_sizes[i][idst] + send_sizes[i+1][idst]));

                if(active[node_rank] == i && ssssize > 0){
                    MPI_Isend(tmp_recv_buffer, ssssize , datatype, 
                            dst, 1, comm, &reqs[num_reqs++]);
                            
                }

                for(int l = 0; l < b; l++){
                    if(active[l] == i){
                        active[( b+ (l - delta * j)) % b] = active[(b+ (l - delta * j)) % b] != i ? i+1 : i;

                        
                        send_sizes[i+1][(b + (l - delta * j)) % b] += send_sizes[i][l];
                        
                        if(j == k-1){
                            active[l] = i+1;
                        }
                    }
                    if(j == k-1){
                        send_sizes[i+1][l] += send_sizes[i][l];
                    }

                }

            }
            MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
            delta *= k;
            
        }

        if(node_rank != 0 && node_rank < nu_count) {
            // Rearrange data in the correct order
            memcpy(recvbuf, 
                    tmp_recv_buffer + ((nu_count) - node_rank) * count * typeSize, 
                    node_rank * count * typeSize);

            memcpy(recvbuf + node_rank * count * typeSize, 
                    tmp_recv_buffer, 
                    ((nu_count) - node_rank) * count * typeSize);

            // memcpy(recvbuf, tmp_recv_buffer, count* (nnodes%b) * typeSize);
            free(tmp_recv_buffer-nstages*b*count*typeSize);
        }

    }else if(node_rank != 0){
        free(tmp_recv_buffer-nstages*b*count*typeSize);
    }


    free(intra_tmp_buf-nstages*count*typeSize);

    free(kgather_tmp_buf);

    free(inter_tmp_buf);

    free(reqs);


    return MPI_SUCCESS;
    



}


#ifdef DEBUG

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* default radix and batch */
    int k = 6, b = 32;
    if (argc >= 3) {
        k = atoi(argv[1]);
        b = atoi(argv[2]);
    }

    if (rank == 0) {
        printf("# allgather_radix_batch vs MPI_Allgather\n");
        printf("# k=%d, b=%d\n", k, b);
        printf("# sendcount   custom_time(s)   std_time(s)\n");
    }

    for (int i = 0; i < 10; ++i) {
        int sendcount = 4 << i;  /* 4,8,16,...,2048 */

        /* allocate buffers as ints; cast to char* for the call */
        int *sendbuf     =(int*)malloc(sendcount * sizeof(int));
        int *recv_custom = (int*)malloc(sendcount * size * sizeof(int));
        int *recv_std    = (int*)malloc(sendcount * size * sizeof(int));
        if (!sendbuf || !recv_custom || !recv_std) {
            fprintf(stderr, "Rank %d: allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* init sendbuf to unique values so we can check correctness */
        for (int j = 0; j < sendcount; ++j) {
            sendbuf[j] = rank * sendcount + j;
        }

        /* synchronize and time custom allgather */
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        allgather_radix_batch((char*)sendbuf,
                              sendcount,
                              MPI_INT,
                              (char*)recv_custom,
                              MPI_COMM_WORLD,
                              k,
                              b);
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        /* synchronize and time standard MPI_Allgather */
        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
        MPI_Allgather(sendbuf,
                      sendcount,
                      MPI_INT,
                      recv_std,
                      sendcount,
                      MPI_INT,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double t3 = MPI_Wtime();

        /* correctness check */
        int errors = 0;
        for (int idx = 0; idx < sendcount * size; ++idx) {
            if (recv_custom[idx] != recv_std[idx]) {
                if (errors < 5 && rank == 0) {
                    int from_rank = idx / sendcount;
                    int offset   = idx % sendcount;
                    fprintf(stderr,
                            "Mismatch at global idx %d (src %d, off %d): custom=%d std=%d\n",
                            idx,
                            from_rank,
                            offset,
                            recv_custom[idx],
                            recv_std[idx]);
                }
                errors++;
            }
        }
        if (errors > 0 && rank == 0) {
            fprintf(stderr, "Total mismatches for sendcount=%d: %d\n",
                    sendcount, errors);
        }

        /* only rank 0 prints the timing results */
        if (rank == 0) {
            printf("%10d   %12.6f   %10.6f\n",
                   sendcount,
                   t1 - t0,
                   t3 - t2);
            fflush(stdout);
        }

        free(sendbuf);
        free(recv_custom);
        free(recv_std);
    }

    MPI_Finalize();
    return 0;
}


#endif