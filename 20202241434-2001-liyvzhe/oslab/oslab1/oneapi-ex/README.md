一，
gemm_basic_ex.cpp实现了对于随机给定的矩阵A(MxK)和B(KxN)（A,B中的每个元素的值是0到1中的随机数）相乘的功能。采取了CPU单线程和GPU的work_group的异构并行计算。GPU中对乘积矩阵C(MxN)进行了分组，每个小组是block_size*block_size的大小。每个work_group并行计算，同时每个work_group中的各个元素也同时计算。

对于gemm_basic_ex.cpp，将26, 27行的全局索引改为当地索引。
当地索引是对于当前的work_group来说的，所以global_id = local_id + group(id)。
由于range是二维的，所以需要指明是哪一维度的索引，就是get_id()相关函数中的parameter。

本程序采用的是传统的CPU单线程O(n^3)的计算结果与GPU的异构并行计算结果相比较来保证GPU计算结果的正确性。同时为了避免因为硬件刚启动时速度较慢，所以先进行了一点次数的预热运行，之后再开始记时。同时通过iterations来进行多次重复计算，求取平均值减小误差。

在原来的github的程序中是没有考虑当M % block_size!=0 || N % block_size!=0的情况，这时候GPU的计算会产生误差。该程序对其进行了小小的优化，使得M或者N不是block_size的时候也能计算正确。


二，
gemm_tile_ex.cpp中利用tileY,tileX创建了subA,subB,sum，分别是临时存储A,B,C中的元素。通过tileY,tileX减少了循环的次数，同时之后再进行异构并行，又一次减小了规模。

对于gemm_tile_ex.cpp。同样，原程序没有考虑tileY，tileX和BLOCK不能整除M和N的情况。这时候会计算错误甚至运行错误，该程序对其进行了小小的优化。而且原来的文件中第44行：
     subA[m] = A[(row + m) * N + k];中的N应该改为K，因为A的每一行有K个元素。
因此，原文件中K和N必须相等才可以，否则会出现错误。

通过tileY，tileX创建了临时的中间矩阵。使得每次循环能够确定更多C的元素。因为临时矩阵的规模远小于C的规模，临时矩阵能减少C的访存次数，因为C是在共享区的一个较大的矩阵，而临时矩阵位于CPU，因此临时矩阵的访问速度快于C的访存速度。类似的，subA,subB也能减少A，B的访存时间。当tileY和tileX远小于A，B的规模时候，tileY,tileX越大，减少A,B,C访问的次数越多，故矩阵计算的性能越高。但是当tileY,tileX的规模接近A，B时，考虑极端情况，tileY,tileX覆盖了A，B，那么此时将变为普通的矩阵乘法，GPU的异构并行失去作用，还多了一个临时矩阵再复制到C的操作，因此性能反而会降低。因此tileY,tileX过大时反而会降低矩阵计算的性能。



三，
对于Cross_Entropy，该文件实现了随机给定的KxMxN的立方体，其中每个值都是(0，1)的随机值，然后一个(0,1)随机值的KxN的mask矩阵，表示在每列中随机选取的元素的行的位置；(0,1)随机值的weight矩阵，表示每列对于计算loss的权重。最终对X做每一个列的softmax归一化处理后形成Y来计算loss[K][N]。当loss的值越小，表明误差越小。

同样，仍类似于（一）（二），采用GPU和CPU的计算结果相对比来确保计算的正确性质，以及通过warmup和iterations来尽可能保证计算时间的稳定性。

本程序的重点在于GPU的并行计算，其中进行了三次并行，第一次是寻找每列的Xmaxi，第二次是计算每列的sum，第三次为了快速，没有将所有的Y的元素都计算出来，只计算了mask中所选取X对应的Y，然后直接带入到loss的计算中。

同时本程序仍考虑了block不能整除K和N的情况，也做了相应的特殊处理。


四，
编译运行。主要可以用两种方式，第一种是通过oneAPI的jupter lab在线编译。首先创建一个终端，然后进入到文件所在的目录。对cpp文件进行dpcpp -o ${filename.cpp} ${targetname}编译。
然后创建一个.sh的脚本文件，例如run.sh，其中的内容是:
     #! /bin/bash
     ./${targetname}

通过qsub -lnodes=1:ppn=2:gpu -d . run.sh 在线运行得到一个run.sh.o和run.sh.e的文件，其中run.sh.o就是运行的结果，run.sh.e是相关的错误报告。

第二种是下载oneAPI base toolkit，然后安装visual studio和intel的显卡驱动程序，安装oneAPI base toolkit的过程中可以直接连接到vs，之后在vs本地运行即可。