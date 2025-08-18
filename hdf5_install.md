# Install HDF5
## Install on Linux/Windows (Ubuntu for example)
```bash
sudo apt update
sudo apt install -y hdf5-tools libhdf5-dev libhdf5-cpp-dev
```


## Install on MacOS (aarch64/arm64)
```bash
## Install homebrew first here: https://brew.sh
brew install hdf5
## then add the below line to your ~/.bash_profile (if you are using bash, change it according for zsh and fish et.al.).
export LDFLAGS="-L/opt/homebrew/opt/hdf5/lib"
export CPPFLAGS="-I/opt/homebrew/opt/hdf5/include"
```
