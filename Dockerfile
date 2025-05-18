FROM julia:1.11

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
	&& apt upgrade -y \
	&& apt install -y \
	build-essential

WORKDIR /usr/local
ENV JULIA_CPU_TARGET=generic

COPY docker/Project.toml .
COPY docker/create_sysimage.jl .

RUN julia --project=. -t auto \
	-O3 \
	--startup-file=no \
	create_sysimage.jl

RUN julia --sysimage /usr/local/vc.so \
	--sysimage-native-code=yes \
	-O3 -t auto \
	--startup-file=no \
	--project=. \
	-e 'using Pkg;Pkg.instantiate()'
