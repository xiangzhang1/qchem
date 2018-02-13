# dask + joblib

        ### Parallel part
        ### =============
        ### if we use 0s for all outs, all takes 10s. thus, speedup is possible.
        ### --------toy run-------------
        ### serial:99s. joblib1:33s. joblib-many:22s.
        ### dask-bag: >5min, it is focused on low-memory not high-performance, it does not feature chunking.
        ### dask-map-gather: 1s-10 res, 24s-1000res, 125s-5000res (pure overhead)
        ### dask-manymap: 165s-5000res-2batch (overhead is relevant to data)
        ### -----------------------------
        ### ---------large toy run---------
        ### preparing data: 90s-18000 items (memory creating, unimportant)
        ### serial: 14s-100 items, 257-1782 items
        ### dask-map-gather: 5s-100 items, 49s-1782 items (startjob-overhead, copymem-overhead)
        ### dask-manymap-gather, 1 batch (~serial): 23s-100 items, 339s-1782 items
        ### dask-manymap-gather, 2 batch: 220s-1782 items (overhead-propto-serial_or_size + comptime / njobs)
        ### dask-manymap-gather, 24 batch: 70s-1782 items (no need for batching?)
        ### dask-bag-map: 37s-1782 items (seems to work with dask.distributed)
        ### --------------------------------
        ### ---------even larger toy run on 2 nodes---------
        ### dask-bag-map, 2threads/2nodes: 471s-1782 items
        ### dask-bag-map, 24processes/1nodes: 46s-1782 items (nprocs work, nthreads don't)
        ### dask-bag-map, 96processes/4nodes: 37s-1782 items
        ### dask-map-gather, 24processes/1nodes: 40s-1782 items
        ### dask-map-gather, 96processes/4nodes: 29s-1782 items
        ### ------------------------------------------------
        ### ---------more processes, multiple nodes---------
        ### dask-map-gather, 96processes, 1500items: 23s
        ### dask-map-gather, 96processes, 5112items: 78s
        ### dask-map-gather, 96processes, 1500items, moreG: 95s
        ### dask-map-gather, 96processes, 5112items, moreG: 323s
        ### joblib24, 1500items, moreG: 51s (too much communication overhead on dask-map)
        ### joblib24, 5112items, moreG: 174s
        ### dask-map-gather, minimal communication overhead (probably), 96processes, 1500items, moreG: 51s
        ### joblib24, 84342items, prodRun: ETA 45min
        ### distributed CPU utilization is very low (~25%). data repacking has limited parallelization.
        ### ---------------------------------------------------------------
        ### -------------rebuilding on irmik-----------------------
        ### joblib, 12 processes, 1500items: 120s (nanaimo 1core = 2.4x irmik 1core)
        ### dask-joblib, 48processes, 1500items: 39s (perfect scaling! wrt time estimate)
        ### dask-joblib, 12*16 processes, 84000items: 759s
        ### dask-joblib, 12*8 processes, 84000items: 880s
        ### ---------------------------------------------------------------
        ### -------------Performance improve--------------------------------------------------
        ### still, parallel performance is sub-par - a lot of time is spent on establishing job pool. batching?
        ### batch-ver 226s
        ### nobatch-ver 765s
        ### 2*batch-ver 192s (good enough!)
        ### dask-map-gather-many (8 nodes, prev=12) > 30min, joblib-dask = 10min
        ### semi-linear scaling... simplicity matters.
        ### ------------------------------------------------------
        ### ------------other--------------------------------------
        ### their model training seems pretty adequate as well.
        ### Always use salloc -N 4 -t 48:00:00 T_T
        ### Instablity (error socket closed) exist, but do not affect computation
        ### ------------------------------------------------------
        ### according to dask, the speedup is only possible if each job is > 100ms. thus, chunking is necessary. Let us try that.

        sys.stderr.write('\n---------- len is %s, selected all, date is %s ----------\n' %(len(ins), str(datetime.now())))
        start = time.time()

        # from dask.distributed import Client
        # import dask.bag as db
        # client = Client('127.0.0.1:8786')
        # print 'client is', client
        # ins = db.from_sequence(ins)
        # outs = ins.map(get_fingerprintprime).compute()

        # from dask.distributed import Client
        # client = Client('localhost:8786')
        # print 'client is', client
        # cores = assign_cores(None, None)
        # n_jobs = np.sum(cores.values())
        # sys.stderr.write('n_jobs is %s\n' %n_jobs)
        # outs_ = client.map(get_fingerprintprime_many, np.array_split(ins, n_jobs * 2))   ### np.array_split is perfect
        # outs  = client.gather(outs_)

        # from joblib import Parallel, delayed
        # outs = Parallel(n_jobs=12)(delayed(get_fingerprintprime)(_) for _ in tqdm(ins))

        import distributed.joblib
        from joblib import Parallel, parallel_backend, delayed
        cores = assign_cores(None, None)
        n_jobs = np.sum(cores.values())
        sys.stderr.write('n_jobs is %s\n' %n_jobs)
        with parallel_backend('dask.distributed', scheduler_host='localhost:8786'):
            outs = Parallel(n_jobs=n_jobs)(delayed(get_fingerprintprime_many)(_) for _ in tqdm(np.array_split(ins, n_jobs*2), leave=False, desc='calculating'))

        end = time.time()
        sys.stderr.write('\n---------- took %s, date is %s ----------\n' %(end-start, str(datetime.now())))
        return dict(zip(keys, outs))


# note: joblib might not blend well with sklearn. it might just hang. there's no workaround.
