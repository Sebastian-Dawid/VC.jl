# VC.jl API Reference

```@contents
```

## VC
```@autodocs
Modules = [VC]
```

## VC.ImageTensorConversion
```@autodocs
Modules = [VC.ImageTensorConversion]
```

## External
You can find the documentation of the most important reexported packages here:
* Linear Algebra: [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
* Machine Learning: [Lux](https://lux.csail.mit.edu/stable/)
* Plotting: [Makie](https://docs.makie.org/stable/)
* Optimization: [Optimisers](https://fluxml.ai/Optimisers.jl/stable/)
* Automatic Differentiation: [Zygote](https://fluxml.ai/Zygote.jl/stable/)

The full reexport looks like this:
```julia
using Reexport
@reexport using FileIO, ImageIO, MeshIO, CairoMakie
@reexport using LinearAlgebra, Statistics, Printf, Random, ProgressMeter
@reexport using StaticArrays, KernelAbstractions
@reexport using Lux, Zygote, Optimisers
@reexport import ColorTypes
```

## Index

```@index
```
