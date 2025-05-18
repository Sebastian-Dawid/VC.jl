using Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.add("PackageCompiler")
Pkg.build("PackageCompiler")
using PackageCompiler

PackageCompiler.create_sysimage(;
    sysimage_path="./vc.so",
    cpu_target="generic",
    sysimage_build_args=`-O3`
)
