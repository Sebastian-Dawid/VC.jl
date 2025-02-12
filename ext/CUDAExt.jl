module CUDAExt

using VC, CUDA

"""
    gpu([T = Float32], arr::AbstractArray)::CuArray{T}

Copies a given array to the GPU.

# Arguments
- `arr`: The array to copy to the GPU.
"""
function VC.gpu(::Type{T}, arr::AbstractArray)::CuArray{T} where {T}
    return CuArray{T}(arr)
end

function __init__()
    VC.GPU_BACKEND = "CUDA"
end

end
