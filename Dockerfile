FROM julia:1.12

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
	&& apt upgrade -y \
	&& apt install -y \
	build-essential

WORKDIR /usr/local
ENV JULIA_CPU_TARGET=generic
ENV JULIA_DEPOT_PATH=/usr/local/.julia

RUN julia -q -O3 -tauto \
	-e 'using Pkg;\
	Pkg.add("PackageCompiler");\
	Pkg.add(url="https://github.com/Sebastian-Dawid/VC.jl.git", rev="v0.2.1");\
	Pkg.add("TestReports");\
	using PackageCompiler;\
	create_sysimage(["VC", "TestReports"];sysimage_path="vc.so", cpu_target="generic", sysimage_build_args=`-O3`)'

CMD [ "julia", "--sysimage=/usr/local/vc.so", "--sysimage-native-code=yes" ]
