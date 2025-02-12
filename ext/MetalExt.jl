module MetalExt

using VC, Metal

"""
    gpu([T = Float32], arr::AbstractArray)::MtlArray{T}

Copies a given array to the GPU.

# Arguments
- `arr`: The array to copy to the GPU.
"""
function VC.gpu(::Type{T}, arr::AbstractArray)::MtlArray{T} where {T}
    return MtlArray{T}(arr)
end

end
