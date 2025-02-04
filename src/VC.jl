module VC

using Preferences, ColorTypes, FileIO
import ImageView

const GPU_BACKEND = @load_preference("gpu_backend", "NONE")

if (GPU_BACKEND == "CUDA")
    using CUDA
elseif (GPU_BACKEND == "AMDGPU")
    using AMDGPU
elseif (GPU_BACKEND == "oneAPI")
    using oneAPI
elseif (GPU_BACKEND == "NONE")
    @warn "No Backend Selected. Using CPU as fallback."
end

to_gpu = !(GPU_BACKEND == "NONE" || GPU_BACKEND == "CPU")

function set_backend(new_backend::String)
    if !(new_backend in ("CUDA", "AMDGPU", "oneAPI", "CPU"))
        throw(ArgumentError("Invalid Backend: \"$new_backend\""))
    end

    @set_preferences!("gpu_backend" => new_backend)
    @info "New backend set; restart your Julia session for this change to take effect!"
end

"""
    gpu(arr::AbstractArray)::AbstractArray

Copies a given array to the GPU based on the selected backend. Has no effect if the "CPU"
or no backend is selected.

# Arguments
- `arr`: The array to copy to the GPU.
"""
function gpu(arr::AbstractArray)::AbstractArray
    @static if (GPU_BACKEND == "CUDA")
        return CuArray(arr)
    elseif (GPU_BACKEND == "AMDGPU")
        return ROCArray(arr)
    elseif (GPU_BACKEND == "oneAPI")
        return oneArray(arr)
    else
        return arr
    end
end


"""
    linspace([T = Float32], start, finish, steps::Integer)::AbstractArray{T, 1} where {T <: AbstractFloat}
"""
function linspace(::Type{T}, start, finish, steps::Integer)::AbstractArray{T, 1} where {T <: AbstractFloat}
    return Array{T}(start:((finish-start)/(steps-1)):finish)
end

function linspace(start, finish, steps::Integer)::AbstractArray{Float32, 1}
    return linspace(Float32, start, finish, steps)
end


"""
    meshgrid(XS::AbstractArray, YS::AbstractArray)::NTuple{2, AbstractArray}
"""
function meshgrid(XS::AbstractArray, YS::AbstractArray)::NTuple{2, AbstractArray}
    X = [ i for i=YS, _=XS ]
    Y = [ j for _=YS, j=XS ]
    return X, Y
end


"""
    tensor([T = Float32], img::AbstractArray{RGB, 2})

Converts an image to a 3xHxW tensor. [`image`](@ref) is the inverse to this function.

See also: [`image`](@ref), [`imshow`](@ref), [`imread`](@ref).

# Arguments
- `T`: Type of the elements of the resulting tensor. Defaults to [`Float32`](@ref).
- `img`: HxW matrix containing the RGB values of the image.

# Example
```jldoctest
julia> img = rand(RGB, 144, 256);

julia> size(img)
(144, 256)

julia> t = img |> tensor;

julia> size(t)
(3, 144, 256)

julia> typeof(t[1])
Float32

julia> t = tensor(Float64, img);

julia> typeof(t[1])
Float64
```
"""
function tensor(::Type{T}, img::AbstractArray{RGB{U}, 2}) where {T <: AbstractFloat, U}
    sz = size(img)
    R = [ T(img[i, j].r) for _=1:1, i=1:sz[1], j=1:sz[2] ]
    G = [ T(img[i, j].g) for _=1:1, i=1:sz[1], j=1:sz[2] ]
    B = [ T(img[i, j].b) for _=1:1, i=1:sz[1], j=1:sz[2] ]
    return cat(R, G, B, dims=1)
end

# use Float32 as the default value for T
function tensor(img::AbstractArray{RGB{U}, 2}) where {U}
    return tensor(Float32, img)
end


"""
    image(t::AbstractArray{T, 3})::AbstractArray{RGB{T}, 2} where {T <: AbstractFloat}

Converts a 3xHxW tensor to a displayable image. [`tensor`](@ref) is the inverse to this function.

See also: [`tensor`](@ref), [`imshow`](@ref), [`imread`](@ref).

# Arguments
- `t`: 3xHxW tensor containing the color data of an image

# Example
```jldoctest
julia> t = rand(3, 144, 256); # image data of an RGB 256x144 image

julia> img = t |> image; # 144x256 matrix containing RGB values

julia> size(img)
(144, 256)

julia> typeof(img[1])
RGB{Float64}
```
"""
function image(tensor::AbstractArray{T, 3})::AbstractArray{RGB{T}, 2} where {T <: AbstractFloat}
    sz = size(tensor)
    return [ RGB(tensor[1, i, j], tensor[2, i, j], tensor[3, i, j]) for i=1:sz[2],j=1:sz[3] ]
end


"""
    imshow(img::AbstractArray{RGB{T}, 2}; show=true, save_to=Nothing) where {T}

Shows and/or saves the given image. Waits until the image is closed.

See also: [`imread`](@ref), [`image`](@ref), [`tensor`](@ref).

# Arguments
- `img`: The image to display and/or save.
- `show`: Determines wheter the image should be displayed.
- `save_to`: Location to save the image to. Defaults to [`Nothing`](@ref) which will not save the image.
"""
function imshow(img::AbstractArray{RGB{T}, 2}; show=true, save_to=Nothing) where {T}
    if (save_to != Nothing)
        save(save_to, img)
    end
    if (show)
        ImageView.imshow(img)
    end
end

function imshow(img::AbstractArray{T, 3}; show=true, save_to=Nothing) where {T <: AbstractFloat}
    imshow(img |> image; show=show, save_to=save_to)
end


"""
    imread([T = Float32], path::String)::AbstractArray{T, 3} where {T <: AbstractFloat}

Loads the given image as a 3xHxW tensor.

See also: [`tensor`](@ref), [`image`](@ref), [`imshow`](@ref)

# Arguments
- `T`: Type of the elements of the resulting tensor. Defaults to [`Float32`](@ref).
- `path`: Path to the image to load.
"""
function imread(::Type{T}, path::String)::AbstractArray{T, 3} where {T <: AbstractFloat}
    img = load(path)
    return tensor(T, img)
end

function imread(path::String)::AbstractArray{Float32, 3}
    return imread(Float32, path)
end

export gpu, linspace, meshgrid, tensor, image, imshow, imread

end # module VC
