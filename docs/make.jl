using Documenter, VC

DocMeta.setdocmeta!(VC, :DocTestSetup, :(using VC); recursive=true)
makedocs(sitename="VC.jl", modules = [VC])
deploydocs(
	repo = "github.com/Sebastian-Dawid/VC.jl.git"
)
