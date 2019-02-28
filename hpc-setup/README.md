# How to access HPC @ DTU

## Connect to HPC thorugh SSH

```
$ ssh userid@login2.gbar.dtu.dk
```

## Get on gpu nodes

### Interactive nodes
Node n-62-18-47 is installed with 1×NVIDIA GForce GTX TITAN X, 2×NVIDIA Tesla K20c, and 1×NVIDIA Tesla K40c, all based on the Kepler architecture (same as NVIDIA Tesla K80c and NVIDIA GForce GTX 680).

To run interactively on this node, you can use the following command:

```
$ k40sh 
```

### LSF10 nodes

2 nodes with 4 x TitanX (Pascal) – queuename: gputitanxpascal  
6 nodes with 2 x Tesla V100 16 GB (owned by DTU Compute&elektro) – queuename: gpuv100  
4 nodes with 2 x Tesla V100 32 GB (owned by DTU Compute&DTU Environment&DTU MEK) – queuename gpuv100  
3 nodes with 4 x Tesla V100 32 GB with NVlink (owned by DTU Compute) – queuename gpuv100  
1 node with 4 x Tesla K80 – queuename: gpuk80  
1 node with 4 x Tesla K40 – queuename: gpuk40  
1 interactive V100-node reachable via voltash  
1 interactive V100-node with NVlink reachable via sxm2sh.

Modify [jobscript.sh](./jobscript.sh) to use one of these nodes

To run jobscript, write `$ bsub > jobscript.sh`

Read more about **bsub** here: [HPC DTU](https://www.hpc.dtu.dk/?page_id=1519)

