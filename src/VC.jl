module VC

using Reexport, Preferences
@reexport using ColorTypes, FileIO
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


"""
    set_backend(new_backend::String)

Sets the GPU backend to the given backend. Requires a restart of Julia to take effect.

# Arguments
- `new_backend`: The name of the new backend.
"""
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

Returns a vector of elements of type `T` from `start` to `finish` in `steps` steps.

# Arguments
- `T`: Type of the elements in the resulting vector. Defaults to [`Float32`](@ref).
- `start`: The first element in the resulting vector.
- `finish`: The last element in the resulting vector.
- `steps`: The number of steps to split the range into.

# Example
```jldoctest
julia> linspace(0, 10, 11)
11-element Vector{Float32}:
  0.0
  1.0
  2.0
  3.0
  4.0
  5.0
  6.0
  7.0
  8.0
  9.0
 10.0
```
"""
function linspace(::Type{T}, start, finish, steps::Integer)::AbstractArray{T, 1} where {T <: AbstractFloat}
    return Array{T}(start:((finish-start)/(steps-1)):finish)
end

function linspace(start, finish, steps::Integer)::AbstractArray{Float32, 1}
    return linspace(Float32, start, finish, steps)
end


"""
    tensor([T = Float32], img::AbstractArray{RGB, 2})::AbstractArray{T, 3}

Converts an image to a CxHxW tensor. [`image`](@ref) is the inverse to this function.

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
function tensor(::Type{T}, img::AbstractArray{U, 2})::AbstractArray{T, 3} where {T <: AbstractFloat, U <: Colorant}
    return mapreduce(x -> reshape(x, (1, size(img)...)), vcat, [ Float32.(@eval($(Symbol("comp$(i)"))).(img)) for i in 1:length(img[1]) ])
end

function tensor(img::AbstractArray{U, 2})::AbstractArray{Float32, 3} where {U <: Colorant}
    return tensor(Float32, img)
end


"""
    image(t::AbstractArray{T, 3})::AbstractArray{U, 2} where {T <: AbstractFloat, U <: Colorant}

Converts a CxHxW tensor to a displayable image. [`tensor`](@ref) is the inverse to this function.

See also: [`tensor`](@ref), [`imshow`](@ref), [`imread`](@ref).

# Arguments
- `t`: CxHxW tensor containing the color data of an image

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
function image(tensor::AbstractArray{T, 3})::AbstractMatrix where {T <: AbstractFloat}
    sz = size(tensor)
    if (sz[1] == 3)
        return [ RGB(tensor[1, i, j], tensor[2, i, j], tensor[3, i, j]) for i=1:sz[2], j=1:sz[3] ]
    else
        return [ Gray(tensor[1, i, j]) for i=1:sz[2], j=1:sz[3] ]
    end
end


"""
    imshow(img::AbstractArray{T, 2}; show=true, save_to=Nothing) where {T <: Colorant}

Shows and/or saves the given image. Waits until the image is closed.

See also: [`imread`](@ref), [`image`](@ref), [`tensor`](@ref).

# Arguments
- `img`: The image to display and/or save.
- `show`: Determines wheter the image should be displayed.
- `save_to`: Location to save the image to. Defaults to [`Nothing`](@ref) which will not save the image.
"""
function imshow(img::AbstractArray{T, 2}; show=true, save_to=Nothing) where {T <: Colorant}
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

Loads the given image as a CxHxW tensor.

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

export gpu, linspace, tensor, image, imshow, imread, set_backend

end # module VC
