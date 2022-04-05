#!/bin/bash

nvidia-smi -pm 1                                                      # enable persistent mode
nvidia-smi -i 0 -pl 140                                                # set power rate limit at 80 watts
nvidia-settings -a "[gpu:0]/GpuPowerMizerMode=2"                      # set performance level 2 (high performance)
#nvidia-settings -a '[gpu:0]/GPUFanControlState=1'                     # set manually controlled fan speed
#nvidia-settings -a '[fan:0]/GPUTargetFanSpeed=75'                     # set fan speed to 75%
nvidia-settings -a '[gpu:0]/GPUGraphicsClockOffset[1]=1000'
nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffset[2]=2000'
nvidia-settings -a '[gpu:0]/GPUGraphicsClockOffset[2]=1000'
nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffset[2]=2000'
nvidia-settings -a '[gpu:0]/GPUGraphicsClockOffset[3]=1000'
nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffset[3]=2000'
nvidia-settings -a '[gpu:0]/GPUGraphicsClockOffset[4]=1000'
nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffset[4]=2000'
