var documenterSearchIndex = {"docs":
[{"location":"#VC.jl-API-Reference","page":"VC.jl API Reference","title":"VC.jl API Reference","text":"","category":"section"},{"location":"","page":"VC.jl API Reference","title":"VC.jl API Reference","text":"","category":"page"},{"location":"#VC","page":"VC.jl API Reference","title":"VC","text":"","category":"section"},{"location":"","page":"VC.jl API Reference","title":"VC.jl API Reference","text":"Modules = [VC]","category":"page"},{"location":"#VC.gpu-Union{Tuple{AbstractArray{T}}, Tuple{T}} where T","page":"VC.jl API Reference","title":"VC.gpu","text":"gpu(arr::AbstractArray{T})::AbstractArray{T} where {T}\n\nThe available GPU backends are:\n\nCUDA (Nvidia)\nAMDGPU (AMD)\noneAPI (Intel)\nMetal (Apple)\n\nThe extension will be loaded when the GPU backend is loaded. E.g.\n\njulia> using CUDA\n\nwill load the CUDA extension.\n\nIf no backend is loaded this function does nothing.\n\n\n\n\n\n","category":"method"},{"location":"#VC.imread-Union{Tuple{T}, Tuple{Type{T}, String}} where T<:AbstractFloat","page":"VC.jl API Reference","title":"VC.imread","text":"imread([T = Float32], path::String)::AbstractArray{T, 3} where {T <: AbstractFloat}\n\nLoads the given image as a HxWxC tensor.\n\nSee also: tensor, ImageTensorConversion.image, imshow\n\nArguments\n\nT: Type of the elements of the resulting tensor. Defaults to Float32.\npath: Path to the image to load.\n\n\n\n\n\n","category":"method"},{"location":"#VC.imshow-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:ColorTypes.Colorant","page":"VC.jl API Reference","title":"VC.imshow","text":"imshow(img::AbstractArray{T, 2}; show=true, save_to=Nothing) where {T <: Colorant}\n\nShows and/or saves the given image. Waits until the image is closed.\n\nSee also: imread, ImageTensorConversion.image, tensor.\n\nArguments\n\nimg: The image to display and/or save.\nshow: Determines wheter the image should be displayed.\nsave_to: Location to save the image to. Defaults to Nothing which will not save the image.\n\n\n\n\n\n","category":"method"},{"location":"#VC.linspace-Union{Tuple{T}, Tuple{Type{T}, Any, Any, Integer}} where T<:AbstractFloat","page":"VC.jl API Reference","title":"VC.linspace","text":"linspace([T = Float32], start, finish, steps::Integer)::AbstractArray{T, 1} where {T <: AbstractFloat}\n\nReturns a vector of elements of type T from start to finish in steps steps.\n\nArguments\n\nT: Type of the elements in the resulting vector. Defaults to Float32.\nstart: The first element in the resulting vector.\nfinish: The last element in the resulting vector.\nsteps: The number of steps to split the range into.\n\nExample\n\njulia> linspace(0, 10, 11)\n11-element Vector{Float32}:\n  0.0\n  1.0\n  2.0\n  3.0\n  4.0\n  5.0\n  6.0\n  7.0\n  8.0\n  9.0\n 10.0\n\n\n\n\n\n","category":"method"},{"location":"#VC.makegrid-Union{Tuple{T}, Tuple{AbstractVecOrMat{<:AbstractArray{T, 3}}, Tuple{var\"#s10\", var\"#s10\"} where var\"#s10\"<:Integer}} where T<:AbstractFloat","page":"VC.jl API Reference","title":"VC.makegrid","text":"makegrid(\n    images::AbstractVecOrMat{<:AbstractArray{T, 3}},\n    dims::NTuple{2, <:Integer}\n    )::AbstractArray{T, 3} where {T <: AbstractFloat}\n\nAranges the images in a grid.\n\nArguments\n\nimages: The images to arange in a grid.\ndims: The dimensions of the grid.\n\nExample\n\njulia> images = [ rand(64, 64, 3) for _ in 1:16 ]; # 16 images of random noise\n\njulia> grid = makegrid(images, (4, 4));\n\njulia> size(grid)\n(256, 256, 3)\n\n\n\n\n\n","category":"method"},{"location":"#VC.row_mul-Union{Tuple{T}, Tuple{AbstractMatrix{T}, AbstractMatrix{T}}} where T<:Real","page":"VC.jl API Reference","title":"VC.row_mul","text":"row_mul(M::AbstractMatrix{T}, vs::AbstractMatrix{T})::AbstractMatrix{T} where {T <: Real}\n\nThis function multiplies the matrix M onto every row of the matrix vs.\n\nArguments\n\nM: The matrix to multiply onto the rows.\nvs: The vectors to mulltiply M onto stored in the rows of a matrix.\n\n\n\n\n\n","category":"method"},{"location":"#VC.show_by_default!-Tuple{Bool}","page":"VC.jl API Reference","title":"VC.show_by_default!","text":"show_by_default!(value::Bool)\n\nSets wether or not to show the image when calling imshow by default.\n\nArguments\n\nvalue: New value for the flag.\n\n\n\n\n\n","category":"method"},{"location":"#VC.ImageTensorConversion","page":"VC.jl API Reference","title":"VC.ImageTensorConversion","text":"","category":"section"},{"location":"","page":"VC.jl API Reference","title":"VC.jl API Reference","text":"Modules = [VC.ImageTensorConversion]","category":"page"},{"location":"#VC.ImageTensorConversion.image-Union{Tuple{AbstractArray{T, 3}}, Tuple{T}} where T<:AbstractFloat","page":"VC.jl API Reference","title":"VC.ImageTensorConversion.image","text":"image(t::AbstractArray{T, 3})::AbstractArray{U, 2} where {T <: AbstractFloat, U <: Colorant}\n\nConverts a HxWxC tensor to a displayable image. tensor is the inverse to this function.\n\nSee also: tensor, VC.imshow, VC.imread.\n\nArguments\n\nt: HxWxC tensor containing the color data of an image\n\nExample\n\njulia> t = rand(144, 256, 3); # image data of an RGB 256x144 image\n\njulia> img = t |> image; # 144x256 matrix containing RGB values\n\njulia> size(img)\n(144, 256)\n\njulia> typeof(img[1])\nRGB{Float64}\n\n\n\n\n\n","category":"method"},{"location":"#VC.ImageTensorConversion.tensor-Union{Tuple{U}, Tuple{T}, Tuple{Type{T}, AbstractMatrix{U}}} where {T<:AbstractFloat, U<:ColorTypes.Colorant}","page":"VC.jl API Reference","title":"VC.ImageTensorConversion.tensor","text":"tensor([T = Float32], img::AbstractArray{RGB, 2})::AbstractArray{T, 3}\n\nConverts an image to a HxWxC tensor. image is the inverse to this function.\n\nSee also: image, VC.imshow, VC.imread.\n\nArguments\n\nT: Type of the elements of the resulting tensor. Defaults to Float32.\nimg: HxW matrix containing the RGB values of the image.\n\nExample\n\njulia> img = rand(RGB, 144, 256);\n\njulia> size(img)\n(144, 256)\n\njulia> t = img |> tensor;\n\njulia> size(t)\n(144, 256, 3)\n\njulia> typeof(t[1])\nFloat32\n\njulia> t = tensor(Float64, img);\n\njulia> typeof(t[1])\nFloat64\n\n\n\n\n\n","category":"method"},{"location":"#External","page":"VC.jl API Reference","title":"External","text":"","category":"section"},{"location":"","page":"VC.jl API Reference","title":"VC.jl API Reference","text":"You can find the documentation of the most important reexported packages here:","category":"page"},{"location":"","page":"VC.jl API Reference","title":"VC.jl API Reference","text":"Linear Algebra: LinearAlgebra\nMachine Learning: Lux\nPlotting: Makie\nOptimization: Optimisers\nAutomatic Differentiation: Zygote","category":"page"},{"location":"","page":"VC.jl API Reference","title":"VC.jl API Reference","text":"The full reexport looks like this:","category":"page"},{"location":"","page":"VC.jl API Reference","title":"VC.jl API Reference","text":"using Reexport\n@reexport using FileIO, ImageIO, MeshIO, CairoMakie\n@reexport using LinearAlgebra, Statistics, Printf, Random, ProgressMeter\n@reexport using StaticArrays, KernelAbstractions\n@reexport using Lux, Zygote, Optimisers\n@reexport import ColorTypes","category":"page"},{"location":"#Index","page":"VC.jl API Reference","title":"Index","text":"","category":"section"},{"location":"","page":"VC.jl API Reference","title":"VC.jl API Reference","text":"","category":"page"}]
}
