if [ "$(uname)" = "Darwin" ]; then
  PROCESSOR_NUM=$(sysctl -n hw.physicalcpu)
elif [ "$(uname)" = "Linux" ]; then
  PROCESSOR_NUM=$(cat /proc/cpuinfo | grep "processor" | wc -l)
fi

export MAX_JOBS=${PROCESSOR_NUM}

pip uninstall chamfer_distance -y

#rm -rf ../chamfer-distance/build
#rm -rf ../chamfer-distance/*.egg-info
#rm ../chamfer-distance/*.so

bear -- python setup.py build_ext --inplace
mv compile_commands.json build

pip install .
