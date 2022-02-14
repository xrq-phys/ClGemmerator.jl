using LinearAlgebra
using OpenCL

const D_TYPE = "float"
const JlType = Float32

const sum_kernel = "
   __kernel void sum(__global const $D_TYPE *a,
                     __global const $D_TYPE *b,
                     __global $D_TYPE *c)
    {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
"
a = rand(JlType, 50000)
b = rand(JlType, 50000)
c = rand(JlType, 50000)

device, ctx, queue = cl.create_compute_context()

a_buff = cl.Buffer(Float32, ctx, (:r, :use), hostbuf=a)
b_buff = cl.Buffer(Float32, ctx, (:r, :use), hostbuf=b)
c_buff = cl.Buffer(Float32, ctx, (:w, :use), hostbuf=c)

p = cl.Program(ctx, source=sum_kernel) |> cl.build!
k = cl.Kernel(p, "sum")

queue(k, size(a), nothing, a_buff, b_buff, c_buff)

# r = cl.read(queue, c_buff)
r = c

if isapprox(norm(r - (a+b)), zero(Float32))
    @info "Success!"
else
    @error "Norm should be 0.0f"
end
