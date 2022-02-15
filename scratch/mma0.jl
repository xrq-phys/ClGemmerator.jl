using LinearAlgebra
using OpenCL

const JlType = Float32
const D_TYPE = "float"

const M_ITEM = 4
const N_ITEM = 4
# const THREADS_M = 8
# const THREADS_N = 4

const F_ = "mma0_$(D_TYPE)_$(M_ITEM)x$(N_ITEM)"
# _$(THREADS_M * M_ITEM)x$(THREADS_N * N_ITEM)_

const REG_NAMES_A = [ "___a_$(i)" for i=1:M_ITEM ]
const REG_NAMES_B = [ "___b_$(j)" for j=1:N_ITEM ]
const REG_NAMES_C = [ "___c_$(i)_$(j)" for i=1:M_ITEM, j=1:N_ITEM ]

const REG_DEFS_A = reduce(string, [ "$D_TYPE $var; " for var=REG_NAMES_A ])
const REG_DEFS_B = reduce(string, [ "$D_TYPE $var; " for var=REG_NAMES_B ])
const REG_DEFS_C = reduce(string, [ "$D_TYPE $var; " for var=REG_NAMES_C ])

const LOAD_VEC_A = reduce(string, [ "$(REG_NAMES_A[i]) = a[$(i-1)]; " for i=1:M_ITEM ])
const LOAD_VEC_B = reduce(string, [ "$(REG_NAMES_B[i]) = b[$(i-1)]; " for i=1:N_ITEM ])

const CLEAN_REGS_C = reduce(string, [ "$var = 0; " for var=REG_NAMES_C ])
const ACCUM_REGS_C = reduce(string, [ "$(REG_NAMES_C[i, j]) += $(REG_NAMES_A[i]) * $(REG_NAMES_B[j]); "
                                     for i=1:M_ITEM, j=1:N_ITEM ])
const FLUSH_REGS_C = reduce(string, [ "c[$(i-1) + $(j-1) * ldc] += $(REG_NAMES_C[i, j]); "
                                     for i=1:M_ITEM, j=1:N_ITEM ])

const mma_kernel_str = "
  __kernel void $F_(int k,
                    __global const $D_TYPE *a, const int psa,
                    __global const $D_TYPE *b, const int psb,
                    __global $D_TYPE *c, const int ldc)
  {
    /*
    // Group id.
    int ig = get_group_id(0);
    int jg = get_group_id(1);

    // Item id.
    int it = get_local_id(0);
    int jt = get_local_id(1);
    */

    int i_ = get_global_id(0);
    int j_ = get_global_id(1);

    // Offset pointers.
    a += i_ * $M_ITEM;
    b += j_ * $N_ITEM;
    c += i_ * $M_ITEM + j_ * $N_ITEM * ldc;

    $REG_DEFS_A
    $REG_DEFS_B
    $REG_DEFS_C

    $CLEAN_REGS_C

    for (int l = 0; l < k; ++l) {
      $LOAD_VEC_A
      $LOAD_VEC_B

      $ACCUM_REGS_C

      a += psa;
      b += psb;
    }

    $FLUSH_REGS_C
  }
"
open("/tmp/ker.cl", "w") do io
    write(io, mma_kernel_str)
end

a = rand(JlType, 2048, 2048);
b = rand(JlType, 2048, 2048);
c = ones(JlType, 2048, 2048);

ctx = cl.create_some_context()
queue = cl.CmdQueue(ctx, :profile)

a_buff = cl.Buffer(JlType, ctx, (:r, :use), hostbuf=a)
b_buff = cl.Buffer(JlType, ctx, (:r, :use), hostbuf=b)
c_buff = cl.Buffer(JlType, ctx, (:w, :use), hostbuf=c)

prog = cl.Program(ctx, source=mma_kernel_str) |> cl.build!
ker = cl.Kernel(prog, F_)

evt = queue(ker, (512, 512), nothing,
            size(a)[2],
            a_buff, strides(a)[2],
            b_buff, strides(b)[2],
            c_buff, strides(c)[2])

@show reduce(max, abs.(c - a * b' .- 1))
@show evt[:profile_duration]
@show 2 * size(a)[1]^3 / evt[:profile_duration] # GFLOPS/sec?

