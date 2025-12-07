#!/bin/bash

#
# This is a helper script for installing the NVIDIA HPC SDK 
# needed to compile a GPU-enabled version of unifrac.
#
# Note: The script currently assumes Linux_x86_64 platform.
# Credit: Igor Sfiligoi (isfiligoi@sdsc.edu), Github handle: sfiligoi

if [ "x${SYSROOT_DIR}" == "x" ]; then
  SYSROOT_DIR=${CONDA_PREFIX}/`$CC -dumpmachine`/sysroot/usr/lib64
fi

# Create GCC symbolic links
# since NVIDIA HPC SDK does not use the env variables
if [ "x${GCC}" == "x" ]; then
  echo "ERROR: GCC not defined"
  exit 1
fi

# usually $CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-
EXE_PREFIX=`echo "$GCC" |sed 's/gcc$//g'`

echo "GCC pointing to ${EXE_PREFIX}gcc"
ls -l ${EXE_PREFIX}gcc

mkdir conda_nv_bins
(cd conda_nv_bins && for f in \
  ar as c++ cc cpp g++ gcc ld nm ranlib gfortran strip; \
  do \
    ln -s ${EXE_PREFIX}${f} ${f}; \
  done )

export PATH=$PWD/conda_nv_bins:$PATH

# Install the NVIDIA HPC SDK

# This link may need to be updated, as new compiler versions are released
# Note: Verified that it works with v25.11
if [ "x${NV_URL}" == "x" ]; then
  if [ "x${NV_ARCH}" == "x" ]; then
    if [ "x`$CC -dumpmachine |grep x86_64`" != "x" ]; then
      NV_ARCH=x86_64
    else
      # else, we guess it is arm
      NV_ARCH=aarch64
    fi
  fi
  NV_URL=https://developer.download.nvidia.com/hpc-sdk/25.11/nvhpc_2025_2511_Linux_${NV_ARCH}_cuda_multi.tar.gz
fi

echo "Downloading the NVIDIA HPC SDK"
# Defaults to using curl
# set USE_CURL=N if you want to use aria2 or wget
if [ "x${USE_CURL}" == "x" ]; then
  # defaults to using inline untarrring
  # set INLINE_CURL=N if you want a temp copy of file on disk
  if [ "x${INLINE_CURL}" == "xN" ]; then
    curl "${NV_URL}" -o nvhpc.tgz
    tar xpzf nvhpc.tgz
    rm -f nvhpc.tgz
  else
    # Do not unpack things we do not use for unifrac
    curl -s "${NV_URL}" | tar xpzf - --exclude '*libcublas*' --exclude '*libcufft*' --exclude '*libcusparse*' --exclude '*libcusolver*' --exclude '*libcurand*' --exclude '*cutensor*' --exclude '*profilers*' --exclude '*comm_libs*' --exclude '*/doc/*' --exclude '*/plugin*'
  fi
elif [ "x${USE_ARIA2}" == "x" ]; then
  aria2c "${NV_URL}"
  tar xpzf nvhpc_*.tar.gz
  rm -f nvhpc_*.tar.gz
else
  wget "${NV_URL}"
  tar xpzf nvhpc_*.tar.gz
  rm -f nvhpc_*.tar.gz
fi

echo "Installing NVIDIA HPC SDK"

# must patch the install scripts to find the right gcc
for f in nvhpc_*/install_components/install nvhpc_*/install_components/*/*/compilers/bin/makelocalrc nvhpc_*/install_components/install_cuda; do
  sed -i -e "s#PATH=/#PATH=$PWD/conda_nv_bins:/#g" $f
done


export NVHPC_INSTALL_DIR=$PWD/hpc_sdk
export NVHPC_SILENT=true

(cd nvhpc_*; ./install)

# create helper scripts
mkdir setup_scripts
cat > setup_scripts/setup_nv_hpc_bins.sh << EOF
PATH=$PWD/conda_nv_bins:`ls -d $PWD/hpc_sdk/*/202*/compilers/bin`:\$PATH

EOF

# patch localrc to find crt1.o
for f in ${NVHPC_INSTALL_DIR}/*/202*/compilers/bin/localrc; do
  echo "set DEFSTDOBJDIR=${SYSROOT_DIR};" >> $f
  #echo "====localrc $f ===="
  #cat $f
  #echo "===="
done

cat > setup_nv_compiler.sh  << EOF
. $PWD/setup_scripts/setup_nv_hpc_bins.sh

. ${PWD}/scripts/enable_nv_compiler.sh

# reuse the same top-level LDFLAGS
# in order to find the right system shared libraries, etc.
export NV_LDFLAGS=\${LDFLAGS}

EOF

# we don't need the install dir anymore
rm -fr nvhpc_*

echo "Setup script avaiabile in $PWD/setup_nv_compiler.sh"
