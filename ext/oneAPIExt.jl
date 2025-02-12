module oneAPIExt

using VC, oneAPI

"""
    gpu([T = Float32], arr::AbstractArray)::oneArray{T}

Copies a given array to the GPU.

# Arguments
- `arr`: The array to copy to the GPU.
"""
function VC.gpu(::Type{T}, arr::AbstractArray)::oneArray{T} where {T}
    return oneArray{T}(arr)
end

end
