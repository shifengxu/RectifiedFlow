
Put these 2 folders in: ~/.cache/torch_extensions.


This is the result files of:
    op/fused_bias_act.cpp
    op/upfirdn2d.cpp
And the above two files (and their related files) will create two *.so files:
    fused.so
    upfirdn2d.so

When makeing models, these two *.so files will be used.