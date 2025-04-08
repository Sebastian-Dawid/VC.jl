module VC

using Reexport
@reexport using FileIO, ImageIO, MeshIO, CairoMakie
@reexport using LinearAlgebra, Statistics, Printf, Random, ProgressMeter
@reexport using StaticArrays, KernelAbstractions
@reexport using Lux, Zygote, Optimisers
@reexport import ColorTypes

module ImageTensorConversion

using ColorTypes

"""
    tensor([T = Float32], img::AbstractArray{RGB, 2})::AbstractArray{T, 3}

Converts an image to a HxWxC tensor. [`image`](@ref) is the inverse to this function.

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
(144, 256, 3)

julia> typeof(t[1])
Float32

julia> t = tensor(Float64, img);

julia> typeof(t[1])
Float64
```
"""
function tensor(::Type{T}, img::AbstractArray{U, 2})::AbstractArray{T, 3} where {T <: AbstractFloat, U <: Colorant}
    return reduce((a, b) -> cat(a, b; dims=3), [ T.(@eval($(Symbol("comp$(i)"))).(img)) for i in 1:length(img[1]) ])
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
julia> t = rand(144, 256, 3); # image data of an RGB 256x144 image

julia> img = t |> image; # 144x256 matrix containing RGB values

julia> size(img)
(144, 256)

julia> typeof(img[1])
RGB{Float64}
```
"""
function image(tensor::AbstractArray{T, 3})::AbstractMatrix where {T <: AbstractFloat}
    sz = size(tensor)
    tensor = clamp.(tensor, 0, 1)
    if (sz[3] == 2)
        return [ GrayA(tensor[i, j, 1], tensor[i, j, 2]) for i=1:sz[1], j=1:sz[2] ]
    elseif (sz[3] == 3)
        return [ RGB(tensor[i, j, 1], tensor[i, j, 2], tensor[i, j, 3]) for i=1:sz[1], j=1:sz[2] ]
    elseif (sz[3] == 4)
        return [ RGBA(tensor[i, j, 1], tensor[i, j, 2], tensor[i, j, 3], tensor[i, j, 4]) for i=1:sz[1], j=1:sz[2] ]
    else
        return [ Gray(tensor[i, j, 1]) for i=1:sz[1], j=1:sz[2] ]
    end
end

export tensor, image

end # module Transform

using .ImageTensorConversion, ColorTypes

GPU_BACKEND::Union{Nothing,String} = nothing
IMAGEVIEW_LOADED::Bool = false
SHOW_BY_DEFAULT::Bool = true

"""
    show_by_default!(value::Bool)

Sets wether or not to show the image when calling [`imshow`](@ref) by default.

# Arguments
- `value`: New value for the flag.
"""
function show_by_default!(value::Bool)
    VC.SHOW_BY_DEFAULT = value
end

"""
    gpu(arr::AbstractArray{T})::AbstractArray{T} where {T}

The available GPU backends are:
- CUDA (Nvidia)
- AMDGPU (AMD)
- oneAPI (Intel)
- Metal (Apple)

The extension will be loaded when the GPU backend is loaded. E.g.
```jldoctest
julia> using CUDA
```
will load the CUDA extension.

If no backend is loaded this function does nothing.
"""
function gpu(arr::AbstractArray{T})::AbstractArray{T} where {T}
    return (isnothing(GPU_BACKEND)) ? arr : gpu(T, arr)
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
    row_mul(M::AbstractMatrix{T}, vs::AbstractMatrix{T})::AbstractMatrix{T} where {T <: Real}

This function multiplies the matrix `M` onto every row of the matrix `vs`.

# Arguments
- `M`: The matrix to multiply onto the rows.
- `vs`: The vectors to mulltiply `M` onto stored in the rows of a matrix.
"""
function row_mul(M::AbstractMatrix{T}, vs::AbstractMatrix{T})::AbstractMatrix{T} where {T <: Real}
    return reduce((a, b) -> cat(a, b; dims=1), map(x -> x' * M', eachrow(vs)))
end


function orthogonalize(M::AbstractMatrix{T})::AbstractMatrix{T} where {T <: AbstractFloat}
	v₁ = normalize(M[:, 1])
	v₃ = normalize(v₁ × M[:, 2])
	return cat(v₁, v₁ × v₃, v₃; dims=2)
end


"""
    makegrid(
        images::AbstractVecOrMat{<:AbstractArray{T, 3}},
        dims::NTuple{2, <:Integer}
        )::AbstractArray{T, 3} where {T <: AbstractFloat}

Aranges the `images` in a grid.

# Arguments
- `images`: The images to arange in a grid.
- `dims`: The dimensions of the grid.

# Example
```jldoctest
julia> images = [ rand(64, 64, 3) for _ in 1:16 ]; # 16 images of random noise
julia> grid = makegrid(images, (4, 4));
julia> size(grid)
(1024, 1024, 3)
```
"""
function makegrid(
    images::AbstractVecOrMat{<:AbstractArray{T, 3}},
    dims::NTuple{2, <:Integer}
    )::AbstractArray{T, 3} where {T <: AbstractFloat}
    grid = reshape(images, dims...)
    return cat([ cat(grid[i, :]...; dims=2) for i in axes(grid, 1) ]...; dims=1)
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
function imshow(img::AbstractArray{T, 2}; show=SHOW_BY_DEFAULT, save_to=Nothing) where {T <: Colorant}
    if (save_to != Nothing)
        save(save_to, img)
    end
    if (show)
        if (IMAGEVIEW_LOADED) # use ImageView backend for displaying images if it is loaded
            imshow(nothing, img)
            return
        end
        f = Figure()
        ax = Axis(f[1, 1])
        hidedecorations!(ax)
        hidespines!(ax)
        CairoMakie.image!(ax, rotr90(img); interpolate=false)
	display(current_figure())
    end
end

function imshow(img::AbstractArray{T, 3}; show=SHOW_BY_DEFAULT, save_to=Nothing) where {T <: AbstractFloat}
    imshow(img |> ImageTensorConversion.image; show=show, save_to=save_to)
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
    return ImageTensorConversion.tensor(T, img)
end

function imread(path::String)::AbstractArray{Float32, 3}
    return imread(Float32, path)
end

export show_by_default!, gpu, linspace, makegrid, row_mul, orthogonalize, imshow, imread, GPU_BACKEND, IMAGEVIEW_LOADED, ImageTensorConversion

end # module VC
