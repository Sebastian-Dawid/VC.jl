module AMDGPUExt

using VC, AMDGPU

"""
    gpu([T = Float32], arr::AbstractArray)::ROCArray{T}

Copies a given array to the GPU.

# Arguments
- `arr`: The array to copy to the GPU.
"""
function VC.gpu(::Type{T}, arr::AbstractArray)::ROCArray{T} where {T}
    return ROCArray{T}(arr)
end

end
